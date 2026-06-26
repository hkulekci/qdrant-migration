package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/pterm/pterm"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/qdrant/go-client/qdrant"

	"github.com/qdrant/migration/pkg/commons"
)

// MigrateToParquetCmd exports a Qdrant collection (vectors + payload + schema) to a
// self-describing Parquet file that can later be imported with the "parquet" command.
type MigrateToParquetCmd struct {
	Qdrant         commons.QdrantConfig    `embed:"" prefix:"qdrant."`
	Parquet        commons.ParquetConfig   `embed:"" prefix:"parquet."`
	Migration      commons.MigrationConfig `embed:"" prefix:"migration."`
	MaxMessageSize int                     `help:"Maximum gRPC message size in bytes (default: 33554432 = 32MB)" default:"33554432" prefix:"qdrant."`

	sourceHost string
	sourcePort int
	sourceTLS  bool
}

func (r *MigrateToParquetCmd) Parse() error {
	var err error
	r.sourceHost, r.sourcePort, r.sourceTLS, err = parseQdrantUrl(r.Qdrant.Url)
	if err != nil {
		return fmt.Errorf("failed to parse source URL: %w", err)
	}
	return nil
}

func (r *MigrateToParquetCmd) Validate() error {
	return validateBatchSize(r.Migration.BatchSize)
}

func (r *MigrateToParquetCmd) Run(globals *Globals) error {
	pterm.DefaultHeader.WithFullWidth().Println("Qdrant to Parquet Export")

	if err := r.Parse(); err != nil {
		return fmt.Errorf("failed to parse input: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	sourceClient, err := connectToQdrant(globals, r.sourceHost, r.sourcePort, r.Qdrant.APIKey, r.sourceTLS, r.MaxMessageSize)
	if err != nil {
		return fmt.Errorf("failed to connect to source: %w", err)
	}
	defer sourceClient.Close()

	collectionInfo, err := sourceClient.GetCollectionInfo(ctx, r.Qdrant.Collection)
	if err != nil {
		return fmt.Errorf("failed to get source collection info: %w", err)
	}

	pointCount, err := sourceClient.Count(ctx, &qdrant.CountPoints{
		CollectionName: r.Qdrant.Collection,
		Exact:          qdrant.PtrOf(true),
	})
	if err != nil {
		return fmt.Errorf("failed to count points in source: %w", err)
	}

	metadata, err := buildExportMetadata(collectionInfo, pointCount)
	if err != nil {
		return fmt.Errorf("failed to build collection metadata: %w", err)
	}

	file, err := os.Create(r.Parquet.Path)
	if err != nil {
		return fmt.Errorf("failed to create parquet file: %w", err)
	}
	defer file.Close()

	writerOpts := []parquet.WriterOption{}
	for k, v := range metadata {
		writerOpts = append(writerOpts, parquet.KeyValueMetadata(k, v))
	}
	writer := parquet.NewGenericWriter[parquetRow](file, writerOpts...)

	displayMigrationStart("qdrant", r.Qdrant.Collection, r.Parquet.Path)

	if err := r.exportData(ctx, sourceClient, writer, pointCount); err != nil {
		_ = writer.Close()
		return fmt.Errorf("failed to export data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to finalize parquet file: %w", err)
	}

	pterm.Success.Printfln("Exported %d points to %s", pointCount, r.Parquet.Path)
	return nil
}

// buildExportMetadata captures the collection's vector and payload schema as footer
// metadata so an import can faithfully recreate the collection.
func buildExportMetadata(info *qdrant.CollectionInfo, pointCount uint64) (map[string]string, error) {
	metadata := map[string]string{
		metaFormatVersion: parquetFormatVersion,
		metaPointCount:    strconv.FormatUint(pointCount, 10),
	}

	marshaler := protojson.MarshalOptions{}
	params := info.GetConfig().GetParams()

	if vc := params.GetVectorsConfig(); vc != nil {
		b, err := marshaler.Marshal(vc)
		if err != nil {
			return nil, fmt.Errorf("failed to encode vectors config: %w", err)
		}
		metadata[metaVectorsConfig] = string(b)
	}

	if svc := params.GetSparseVectorsConfig(); svc != nil {
		b, err := marshaler.Marshal(svc)
		if err != nil {
			return nil, fmt.Errorf("failed to encode sparse vectors config: %w", err)
		}
		metadata[metaSparseVectorsConfig] = string(b)
	}

	if schema := info.GetPayloadSchema(); len(schema) > 0 {
		encoded := make(map[string]string, len(schema))
		for field, schemaInfo := range schema {
			b, err := marshaler.Marshal(schemaInfo)
			if err != nil {
				return nil, fmt.Errorf("failed to encode payload schema for %q: %w", field, err)
			}
			encoded[field] = string(b)
		}
		b, err := json.Marshal(encoded)
		if err != nil {
			return nil, fmt.Errorf("failed to encode payload schema: %w", err)
		}
		metadata[metaPayloadSchema] = string(b)
	}

	return metadata, nil
}

// exportData scrolls through the source collection and writes each point as a row.
func (r *MigrateToParquetCmd) exportData(ctx context.Context, sourceClient *qdrant.Client, writer *parquet.GenericWriter[parquetRow], pointCount uint64) error {
	bar, _ := pterm.DefaultProgressbar.WithTotal(int(pointCount)).Start()

	var offset *qdrant.PointId
	for {
		resp, err := sourceClient.GetPointsClient().Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: r.Qdrant.Collection,
			Offset:         offset,
			Limit:          qdrant.PtrOf(uint32(r.Migration.BatchSize)),
			WithPayload:    qdrant.NewWithPayload(true),
			WithVectors:    qdrant.NewWithVectors(true),
		})
		if err != nil {
			return fmt.Errorf("failed to scroll data from source: %w", err)
		}

		points := resp.GetResult()
		rows := make([]parquetRow, 0, len(points))
		for _, p := range points {
			vectors, err := retrievedVectorsToJSON(p)
			if err != nil {
				return fmt.Errorf("point %s: %w", pointIDToString(p.Id), err)
			}
			payload, err := payloadToJSON(p.Payload)
			if err != nil {
				return fmt.Errorf("point %s: %w", pointIDToString(p.Id), err)
			}
			rows = append(rows, parquetRow{
				ID:      pointIDToString(p.Id),
				Vectors: vectors,
				Payload: payload,
			})
		}

		if len(rows) > 0 {
			if _, err := writer.Write(rows); err != nil {
				return fmt.Errorf("failed to write rows: %w", err)
			}
		}

		bar.Add(len(points))
		offset = resp.GetNextPageOffset()
		if offset == nil {
			break
		}

		if r.Migration.BatchDelay > 0 {
			time.Sleep(time.Duration(r.Migration.BatchDelay) * time.Millisecond)
		}
	}

	pterm.Success.Printfln("Data export finished successfully")
	return nil
}
