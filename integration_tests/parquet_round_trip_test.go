package integrationtests

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/qdrant/go-client/qdrant"
)

// TestParquetRoundTrip exports a Qdrant collection to a Parquet file and imports it
// back into a fresh Qdrant instance, asserting that vectors (named dense + sparse),
// typed payload values, and the payload index all survive the round trip.
func TestParquetRoundTrip(t *testing.T) {
	ctx := context.Background()

	sourceContainer := qdrantContainer(ctx, t, qdrantAPIKey)
	defer func() {
		if err := sourceContainer.Terminate(ctx); err != nil {
			t.Errorf("Failed to terminate source Qdrant container: %v", err)
		}
	}()

	targetContainer := qdrantContainer(ctx, t, qdrantAPIKey)
	defer func() {
		if err := targetContainer.Terminate(ctx); err != nil {
			t.Errorf("Failed to terminate target Qdrant container: %v", err)
		}
	}()

	sourceHost, err := sourceContainer.Host(ctx)
	require.NoError(t, err)
	sourcePort, err := sourceContainer.MappedPort(ctx, qdrantGRPCPort)
	require.NoError(t, err)

	targetHost, err := targetContainer.Host(ctx)
	require.NoError(t, err)
	targetPort, err := targetContainer.MappedPort(ctx, qdrantGRPCPort)
	require.NoError(t, err)

	sourceClient, err := qdrant.NewClient(&qdrant.Config{
		Host:                   sourceHost,
		Port:                   sourcePort.Int(),
		APIKey:                 qdrantAPIKey,
		SkipCompatibilityCheck: true,
	})
	require.NoError(t, err)
	defer sourceClient.Close()

	targetClient, err := qdrant.NewClient(&qdrant.Config{
		Host:                   targetHost,
		Port:                   targetPort.Int(),
		APIKey:                 qdrantAPIKey,
		SkipCompatibilityCheck: true,
	})
	require.NoError(t, err)
	defer targetClient.Close()

	const sourceCollectionName = "parquet_source"
	const targetCollectionName = "parquet_target"

	// Source collection with a named dense vector and a sparse vector.
	// Dot distance is used so dense vectors are not normalized and can be compared exactly.
	err = sourceClient.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: sourceCollectionName,
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			"dense": {Size: dimension, Distance: qdrant.Distance_Dot},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			"sparse": {},
		}),
	})
	require.NoError(t, err)

	_, err = sourceClient.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
		CollectionName: sourceCollectionName,
		FieldName:      "category",
		FieldType:      qdrant.PtrOf(qdrant.FieldType_FieldTypeKeyword),
		Wait:           qdrant.PtrOf(true),
	})
	require.NoError(t, err)

	type expectedPoint struct {
		dense        []float32
		sparseIdx    []uint32
		sparseValues []float32
		category     string
		count        int64
		score        float64
		active       bool
	}
	expected := make(map[string]expectedPoint, totalEntries)

	points := make([]*qdrant.PointStruct, 0, totalEntries)
	for i := 0; i < totalEntries; i++ {
		id := uuid.New().String()
		dense := randFloat32Values(dimension)
		sparseIdx := []uint32{0, 2, 5}
		sparseValues := randFloat32Values(3)
		exp := expectedPoint{
			dense:        dense,
			sparseIdx:    sparseIdx,
			sparseValues: sparseValues,
			category:     fmt.Sprintf("cat-%d", i%3),
			count:        int64(i),
			score:        float64(i) + 0.5,
			active:       i%2 == 0,
		}
		expected[id] = exp

		points = append(points, &qdrant.PointStruct{
			Id: qdrant.NewID(id),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense":  qdrant.NewVectorDense(dense),
				"sparse": qdrant.NewVectorSparse(sparseIdx, sparseValues),
			}),
			Payload: qdrant.NewValueMap(map[string]any{
				"category": exp.category,
				"count":    exp.count,
				"score":    exp.score,
				"active":   exp.active,
			}),
		})
	}

	_, err = sourceClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: sourceCollectionName,
		Points:         points,
		Wait:           qdrant.PtrOf(true),
	})
	require.NoError(t, err)

	parquetPath := filepath.Join(t.TempDir(), "round_trip.parquet")

	// Export the source collection to a Parquet file.
	runMigrationBinary(t, []string{
		"to-parquet",
		fmt.Sprintf("--qdrant.url=http://%s:%s", sourceHost, sourcePort.Port()),
		fmt.Sprintf("--qdrant.api-key=%s", qdrantAPIKey),
		fmt.Sprintf("--qdrant.collection=%s", sourceCollectionName),
		fmt.Sprintf("--parquet.path=%s", parquetPath),
	})

	// Import the Parquet file into a fresh target collection.
	runMigrationBinary(t, []string{
		"parquet",
		fmt.Sprintf("--parquet.path=%s", parquetPath),
		fmt.Sprintf("--qdrant.url=http://%s:%s", targetHost, targetPort.Port()),
		fmt.Sprintf("--qdrant.api-key=%s", qdrantAPIKey),
		fmt.Sprintf("--qdrant.collection=%s", targetCollectionName),
		"--migration.create-collection=true",
	})

	// The target collection must have been recreated with both vector kinds and the index.
	targetInfo, err := targetClient.GetCollectionInfo(ctx, targetCollectionName)
	require.NoError(t, err)
	require.Contains(t, targetInfo.GetConfig().GetParams().GetVectorsConfig().GetParamsMap().GetMap(), "dense")
	require.Contains(t, targetInfo.GetConfig().GetParams().GetSparseVectorsConfig().GetMap(), "sparse")
	require.Contains(t, targetInfo.GetPayloadSchema(), "category")

	count, err := targetClient.Count(ctx, &qdrant.CountPoints{
		CollectionName: targetCollectionName,
		Exact:          qdrant.PtrOf(true),
	})
	require.NoError(t, err)
	require.Equal(t, uint64(totalEntries), count)

	migrated, err := targetClient.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: targetCollectionName,
		Limit:          qdrant.PtrOf(uint32(totalEntries)),
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	require.NoError(t, err)
	require.Len(t, migrated, totalEntries)

	for _, p := range migrated {
		exp, ok := expected[p.Id.GetUuid()]
		require.True(t, ok, "unexpected point id %s", p.Id.GetUuid())

		named := p.Vectors.GetVectors().GetVectors()

		// Dense vector preserved exactly.
		require.Equal(t, exp.dense, named["dense"].GetData())

		// Sparse vector indices and values preserved.
		require.Equal(t, exp.sparseIdx, named["sparse"].GetIndices().GetData())
		require.Equal(t, exp.sparseValues, named["sparse"].GetData())

		// Payload values keep their original types.
		require.Equal(t, exp.category, p.Payload["category"].GetStringValue())
		require.Equal(t, exp.count, p.Payload["count"].GetIntegerValue())
		require.Equal(t, exp.score, p.Payload["score"].GetDoubleValue())
		require.Equal(t, exp.active, p.Payload["active"].GetBoolValue())

		// Integers must not have been coerced into doubles.
		_, isInt := p.Payload["count"].GetKind().(*qdrant.Value_IntegerValue)
		require.True(t, isInt, "count payload should be an integer value")
	}
}
