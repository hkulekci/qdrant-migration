package cmd

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/parquet-go/parquet-go"
	"github.com/qdrant/go-client/qdrant"
)

func TestPointIDRoundTrip(t *testing.T) {
	cases := []*qdrant.PointId{
		qdrant.NewIDNum(0),
		qdrant.NewIDNum(42),
		qdrant.NewIDNum(18446744073709551615), // max uint64
		qdrant.NewIDUUID("550e8400-e29b-41d4-a716-446655440000"),
	}
	for _, id := range cases {
		got := stringToPointID(pointIDToString(id))
		if pointIDToString(got) != pointIDToString(id) {
			t.Errorf("point ID round-trip mismatch: %v -> %q -> %v", id, pointIDToString(id), got)
		}
	}
}

func TestPayloadRoundTrip(t *testing.T) {
	payload := qdrant.NewValueMap(map[string]any{
		"title":  "hello",
		"count":  int64(7),
		"score":  3.5,
		"active": true,
		"nested": map[string]any{"a": int64(1), "b": "two"},
		"tags":   []any{"x", "y", int64(3)},
	})
	// Explicit null which NewValueMap does not produce from a typed map.
	payload["empty"] = qdrant.NewValueNull()

	encoded, err := payloadToJSON(payload)
	if err != nil {
		t.Fatalf("payloadToJSON: %v", err)
	}
	decoded, err := jsonToPayload(encoded)
	if err != nil {
		t.Fatalf("jsonToPayload: %v", err)
	}

	if !reflect.DeepEqual(valueToAny(qdrant.NewValueFromFields(payload)), valueToAny(qdrant.NewValueFromFields(decoded))) {
		t.Errorf("payload round-trip mismatch:\n in:  %v\n out: %v", payload, decoded)
	}

	// Integers must stay integers, not become floats.
	if _, ok := decoded["count"].GetKind().(*qdrant.Value_IntegerValue); !ok {
		t.Errorf("integer payload value lost its type: %T", decoded["count"].GetKind())
	}
	if _, ok := decoded["score"].GetKind().(*qdrant.Value_DoubleValue); !ok {
		t.Errorf("float payload value lost its type: %T", decoded["score"].GetKind())
	}
}

func TestEmptyPayloadRoundTrip(t *testing.T) {
	encoded, err := payloadToJSON(nil)
	if err != nil {
		t.Fatalf("payloadToJSON(nil): %v", err)
	}
	if encoded != "" {
		t.Errorf("expected empty string for empty payload, got %q", encoded)
	}
	decoded, err := jsonToPayload(encoded)
	if err != nil {
		t.Fatalf("jsonToPayload(\"\"): %v", err)
	}
	if decoded != nil {
		t.Errorf("expected nil payload, got %v", decoded)
	}
}

func TestVectorsJSONRoundTrip(t *testing.T) {
	cases := map[string]vectorsJSON{
		"single dense": {Single: &vectorJSON{Dense: []float32{0.1, 0.2, 0.3}}},
		"named mixed": {Named: map[string]vectorJSON{
			"dense":  {Dense: []float32{1, 2, 3, 4}},
			"sparse": {Sparse: &sparseJSON{Indices: []uint32{0, 5, 9}, Values: []float32{0.5, 1.5, 2.5}}},
			"multi":  {Multi: [][]float32{{1, 2}, {3, 4}, {5, 6}}},
		}},
	}

	for name, vs := range cases {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(vs)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			vectors, err := jsonToVectors(string(encoded))
			if err != nil {
				t.Fatalf("jsonToVectors: %v", err)
			}
			if vectors == nil {
				t.Fatal("expected non-nil vectors")
			}
		})
	}
}

// TestParquetFileRoundTrip writes rows plus footer metadata to a real parquet file,
// reopens it, and verifies both the metadata and the row data survive the trip.
func TestParquetFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.parquet")

	want := []parquetRow{
		{ID: "1", Vectors: `{"single":{"dense":[0.1,0.2]}}`, Payload: `{"a":1}`},
		{ID: "550e8400-e29b-41d4-a716-446655440000", Vectors: `{"named":{"s":{"sparse":{"indices":[0,3],"values":[1,2]}}}}`, Payload: ""},
	}
	meta := map[string]string{
		metaFormatVersion: parquetFormatVersion,
		metaPointCount:    "2",
		metaVectorsConfig: `{"params":{"size":"2","distance":"Cosine"}}`,
	}

	out, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	opts := []parquet.WriterOption{}
	for k, v := range meta {
		opts = append(opts, parquet.KeyValueMetadata(k, v))
	}
	w := parquet.NewGenericWriter[parquetRow](out, opts...)
	if _, err := w.Write(want); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("writer close: %v", err)
	}
	if err := out.Close(); err != nil {
		t.Fatalf("file close: %v", err)
	}

	in, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer in.Close()
	stat, _ := in.Stat()

	pqFile, err := parquet.OpenFile(in, stat.Size())
	if err != nil {
		t.Fatalf("open parquet: %v", err)
	}
	for k, v := range meta {
		got, ok := pqFile.Lookup(k)
		if !ok || got != v {
			t.Errorf("metadata %q = %q (ok=%v), want %q", k, got, ok, v)
		}
	}
	if pqFile.NumRows() != int64(len(want)) {
		t.Errorf("NumRows = %d, want %d", pqFile.NumRows(), len(want))
	}

	r := parquet.NewGenericReader[parquetRow](in)
	defer r.Close()
	got := make([]parquetRow, len(want))
	n, err := r.Read(got)
	if err != nil && err != io.EOF {
		t.Fatalf("read: %v", err)
	}
	if n != len(want) {
		t.Fatalf("read %d rows, want %d", n, len(want))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("rows mismatch:\n got:  %v\n want: %v", got, want)
	}

	// The recovered rows must rebuild into valid points.
	if _, err := rowsToPoints(got); err != nil {
		t.Errorf("rowsToPoints: %v", err)
	}
}
