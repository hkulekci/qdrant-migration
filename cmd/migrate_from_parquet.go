package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/pterm/pterm"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/qdrant/go-client/qdrant"

	"github.com/qdrant/migration/pkg/commons"
)

// MigrateFromParquetCmd imports a Parquet file produced by the "to-parquet" command
// (or any file following the same layout) into a Qdrant collection.
type MigrateFromParquetCmd struct {
	Parquet   commons.ParquetConfig   `embed:"" prefix:"parquet."`
	Qdrant    commons.QdrantConfig    `embed:"" prefix:"qdrant."`
	Migration commons.MigrationConfig `embed:"" prefix:"migration."`

	targetHost string
	targetPort int
	targetTLS  bool
}

func (r *MigrateFromParquetCmd) Parse() error {
	var err error
	r.targetHost, r.targetPort, r.targetTLS, err = parseQdrantUrl(r.Qdrant.Url)
	if err != nil {
		return fmt.Errorf("failed to parse target URL: %w", err)
	}
	return nil
}

func (r *MigrateFromParquetCmd) Validate() error {
	return validateBatchSize(r.Migration.BatchSize)
}

func (r *MigrateFromParquetCmd) Run(globals *Globals) error {
	pterm.DefaultHeader.WithFullWidth().Println("Parquet to Qdrant Data Migration")

	if err := r.Parse(); err != nil {
		return fmt.Errorf("failed to parse input: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	file, err := os.Open(r.Parquet.Path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat parquet file: %w", err)
	}

	pqFile, err := parquet.OpenFile(file, stat.Size())
	if err != nil {
		return fmt.Errorf("failed to read parquet file: %w", err)
	}

	targetClient, err := connectToQdrant(globals, r.targetHost, r.targetPort, r.Qdrant.APIKey, r.targetTLS, 0)
	if err != nil {
		return fmt.Errorf("failed to connect to Qdrant target: %w", err)
	}
	defer targetClient.Close()

	if err := commons.PrepareOffsetsCollection(ctx, r.Migration.OffsetsCollection, targetClient); err != nil {
		return fmt.Errorf("failed to prepare migration marker collection: %w", err)
	}

	if err := r.prepareTargetCollection(ctx, pqFile, targetClient); err != nil {
		return fmt.Errorf("error preparing target collection: %w", err)
	}

	totalRows := pqFile.NumRows()
	displayMigrationStart("parquet", r.Parquet.Path, r.Qdrant.Collection)

	if err := r.importData(ctx, file, targetClient, totalRows); err != nil {
		return fmt.Errorf("failed to import data: %w", err)
	}

	targetPointCount, err := targetClient.Count(ctx, &qdrant.CountPoints{
		CollectionName: r.Qdrant.Collection,
		Exact:          qdrant.PtrOf(true),
	})
	if err != nil {
		return fmt.Errorf("failed to count points in target: %w", err)
	}
	pterm.Info.Printfln("Target collection has %d points\n", targetPointCount)

	if err := commons.DeleteOffsetsCollection(ctx, r.Migration.OffsetsCollection, targetClient); err != nil {
		return fmt.Errorf("failed to delete migration marker collection: %w", err)
	}

	return nil
}

// prepareTargetCollection recreates the collection from the footer metadata when it
// does not already exist, including vector configuration and payload indexes.
func (r *MigrateFromParquetCmd) prepareTargetCollection(ctx context.Context, pqFile *parquet.File, targetClient *qdrant.Client) error {
	if !r.Migration.CreateCollection {
		return nil
	}

	exists, err := targetClient.CollectionExists(ctx, r.Qdrant.Collection)
	if err != nil {
		return fmt.Errorf("failed to check if collection exists: %w", err)
	}
	if exists {
		pterm.Info.Printfln("Target collection '%s' already exists. Skipping creation.", r.Qdrant.Collection)
		return nil
	}

	createReq := &qdrant.CreateCollection{CollectionName: r.Qdrant.Collection}

	if raw, ok := pqFile.Lookup(metaVectorsConfig); ok {
		var vc qdrant.VectorsConfig
		if err := protojson.Unmarshal([]byte(raw), &vc); err != nil {
			return fmt.Errorf("failed to decode vectors config: %w", err)
		}
		createReq.VectorsConfig = &vc
	}
	if raw, ok := pqFile.Lookup(metaSparseVectorsConfig); ok {
		var svc qdrant.SparseVectorConfig
		if err := protojson.Unmarshal([]byte(raw), &svc); err != nil {
			return fmt.Errorf("failed to decode sparse vectors config: %w", err)
		}
		createReq.SparseVectorsConfig = &svc
	}

	if createReq.VectorsConfig == nil && createReq.SparseVectorsConfig == nil {
		return fmt.Errorf("parquet file has no vector configuration metadata; cannot create collection (use --migration.create-collection=false to import into an existing collection)")
	}

	if err := targetClient.CreateCollection(ctx, createReq); err != nil {
		return fmt.Errorf("failed to create target collection: %w", err)
	}
	pterm.Success.Printfln("Created target collection '%s'", r.Qdrant.Collection)

	return r.createPayloadIndexes(ctx, pqFile, targetClient)
}

// createPayloadIndexes recreates payload field indexes recorded in the footer metadata.
func (r *MigrateFromParquetCmd) createPayloadIndexes(ctx context.Context, pqFile *parquet.File, targetClient *qdrant.Client) error {
	raw, ok := pqFile.Lookup(metaPayloadSchema)
	if !ok {
		return nil
	}

	var encoded map[string]string
	if err := json.Unmarshal([]byte(raw), &encoded); err != nil {
		return fmt.Errorf("failed to decode payload schema: %w", err)
	}

	for field, schemaJSON := range encoded {
		var schemaInfo qdrant.PayloadSchemaInfo
		if err := protojson.Unmarshal([]byte(schemaJSON), &schemaInfo); err != nil {
			return fmt.Errorf("failed to decode payload schema for %q: %w", field, err)
		}
		fieldType := getFieldType(schemaInfo.GetDataType())
		if fieldType == nil {
			continue
		}
		if _, err := targetClient.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName:   r.Qdrant.Collection,
			FieldName:        field,
			FieldType:        fieldType,
			FieldIndexParams: schemaInfo.GetParams(),
			Wait:             qdrant.PtrOf(true),
		}); err != nil {
			return fmt.Errorf("failed creating index on field %q: %w", field, err)
		}
	}
	return nil
}

// importData reads rows from the parquet file in batches and upserts them, resuming
// from the stored offset when not restarting.
func (r *MigrateFromParquetCmd) importData(ctx context.Context, file *os.File, targetClient *qdrant.Client, totalRows int64) error {
	reader := parquet.NewGenericReader[parquetRow](file)
	defer reader.Close()

	var imported uint64
	if !r.Migration.Restart {
		_, stored, err := commons.GetStartOffset(ctx, r.Migration.OffsetsCollection, targetClient, r.Parquet.Path)
		if err != nil {
			return fmt.Errorf("failed to get start offset: %w", err)
		}
		imported = stored
		if imported > 0 {
			if err := reader.SeekToRow(int64(imported)); err != nil {
				return fmt.Errorf("failed to seek to offset %d: %w", imported, err)
			}
		}
	}

	bar, _ := pterm.DefaultProgressbar.WithTotal(int(totalRows)).Start()
	displayMigrationProgress(bar, imported)

	buf := make([]parquetRow, r.Migration.BatchSize)
	for {
		n, readErr := reader.Read(buf)
		if n > 0 {
			points, err := rowsToPoints(buf[:n])
			if err != nil {
				return err
			}
			if err := upsertWithRetry(ctx, targetClient, &qdrant.UpsertPoints{
				CollectionName: r.Qdrant.Collection,
				Points:         points,
				Wait:           qdrant.PtrOf(true),
			}); err != nil {
				return err
			}

			imported += uint64(n)
			bar.Add(n)

			// A fixed placeholder ID is used; only the offset count matters for resuming.
			if err := commons.StoreStartOffset(ctx, r.Migration.OffsetsCollection, targetClient, r.Parquet.Path, qdrant.NewIDNum(0), imported); err != nil {
				return fmt.Errorf("failed to store offset: %w", err)
			}

			if r.Migration.BatchDelay > 0 {
				time.Sleep(time.Duration(r.Migration.BatchDelay) * time.Millisecond)
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return fmt.Errorf("failed to read parquet rows: %w", readErr)
		}
	}

	pterm.Success.Printfln("Data migration finished successfully")
	return nil
}

// rowsToPoints converts parquet rows into upsert-ready Qdrant points.
func rowsToPoints(rows []parquetRow) ([]*qdrant.PointStruct, error) {
	points := make([]*qdrant.PointStruct, 0, len(rows))
	for _, row := range rows {
		vectors, err := jsonToVectors(row.Vectors)
		if err != nil {
			return nil, fmt.Errorf("point %s: %w", row.ID, err)
		}
		payload, err := jsonToPayload(row.Payload)
		if err != nil {
			return nil, fmt.Errorf("point %s: %w", row.ID, err)
		}
		points = append(points, &qdrant.PointStruct{
			Id:      stringToPointID(row.ID),
			Vectors: vectors,
			Payload: payload,
		})
	}
	return points, nil
}
