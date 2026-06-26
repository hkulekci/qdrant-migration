package cmd

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/qdrant/go-client/qdrant"
)

// Footer (file-level) metadata keys used to make a Parquet export self-describing,
// so that an import can recreate the collection with the same vector and payload schema.
const (
	// parquetFormatVersion is bumped when the on-disk layout changes incompatibly.
	parquetFormatVersion = "1"

	metaFormatVersion       = "qdrant_migrate.format_version"
	metaVectorsConfig       = "qdrant_migrate.vectors_config"        // protojson of *qdrant.VectorsConfig
	metaSparseVectorsConfig = "qdrant_migrate.sparse_vectors_config" // protojson of *qdrant.SparseVectorConfig
	metaPayloadSchema       = "qdrant_migrate.payload_schema"        // JSON map[field]protojson(*qdrant.PayloadSchemaInfo)
	metaPointCount          = "qdrant_migrate.point_count"
)

// parquetRow is a single stored point. Vectors and payload are kept as JSON strings
// so the format round-trips losslessly across all vector kinds (dense, named, sparse,
// multi) and the heterogeneous, typed payload values Qdrant allows.
type parquetRow struct {
	ID      string `parquet:"id"`
	Vectors string `parquet:"vectors"`
	Payload string `parquet:"payload"`
}

// sparseJSON is the JSON shape of a sparse vector.
type sparseJSON struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// vectorJSON holds exactly one vector kind. The set field disambiguates dense vs
// sparse vs multi-dense on import.
type vectorJSON struct {
	Dense  []float32   `json:"dense,omitempty"`
	Sparse *sparseJSON `json:"sparse,omitempty"`
	Multi  [][]float32 `json:"multi,omitempty"`
}

// vectorsJSON is the per-point vector container: either a single unnamed vector or a
// map of named vectors, mirroring Qdrant's VectorsOutput.
type vectorsJSON struct {
	Single *vectorJSON           `json:"single,omitempty"`
	Named  map[string]vectorJSON `json:"named,omitempty"`
}

// ---------------------------------------------------------------------------
// Point ID <-> string
// ---------------------------------------------------------------------------

// pointIDToString renders a point ID as a string. Numeric IDs become plain digits
// and UUIDs are stored verbatim; the two are unambiguous since a UUID always
// contains non-digit characters.
func pointIDToString(id *qdrant.PointId) string {
	if id == nil {
		return ""
	}
	if uuidStr := id.GetUuid(); uuidStr != "" {
		return uuidStr
	}
	return strconv.FormatUint(id.GetNum(), 10)
}

// stringToPointID reverses pointIDToString. An all-digit value that fits in a uint64
// is treated as a numeric ID, everything else as a UUID.
func stringToPointID(s string) *qdrant.PointId {
	if n, err := strconv.ParseUint(s, 10, 64); err == nil {
		return qdrant.NewIDNum(n)
	}
	return qdrant.NewIDUUID(s)
}

// ---------------------------------------------------------------------------
// Payload Value <-> plain Go value (for JSON)
// ---------------------------------------------------------------------------

// valueToAny converts a Qdrant payload Value into a plain Go value suitable for JSON
// encoding, preserving the integer/float distinction.
func valueToAny(v *qdrant.Value) any {
	if v == nil {
		return nil
	}
	switch k := v.GetKind().(type) {
	case *qdrant.Value_NullValue:
		return nil
	case *qdrant.Value_BoolValue:
		return k.BoolValue
	case *qdrant.Value_IntegerValue:
		return k.IntegerValue
	case *qdrant.Value_DoubleValue:
		return k.DoubleValue
	case *qdrant.Value_StringValue:
		return k.StringValue
	case *qdrant.Value_StructValue:
		fields := k.StructValue.GetFields()
		m := make(map[string]any, len(fields))
		for key, val := range fields {
			m[key] = valueToAny(val)
		}
		return m
	case *qdrant.Value_ListValue:
		vals := k.ListValue.GetValues()
		arr := make([]any, len(vals))
		for i, val := range vals {
			arr[i] = valueToAny(val)
		}
		return arr
	}
	return nil
}

// anyToValue converts a JSON value decoded with json.Number into a Qdrant Value,
// keeping integers and floating-point numbers distinct.
func anyToValue(v any) (*qdrant.Value, error) {
	switch t := v.(type) {
	case nil:
		return qdrant.NewValueNull(), nil
	case bool:
		return qdrant.NewValueBool(t), nil
	case string:
		return qdrant.NewValueString(t), nil
	case json.Number:
		if !strings.ContainsAny(t.String(), ".eE") {
			if i, err := t.Int64(); err == nil {
				return qdrant.NewValueInt(i), nil
			}
		}
		f, err := t.Float64()
		if err != nil {
			return nil, fmt.Errorf("invalid number %q: %w", t.String(), err)
		}
		return qdrant.NewValueDouble(f), nil
	case float64:
		return qdrant.NewValueDouble(t), nil
	case map[string]any:
		fields := make(map[string]*qdrant.Value, len(t))
		for key, val := range t {
			cv, err := anyToValue(val)
			if err != nil {
				return nil, err
			}
			fields[key] = cv
		}
		return qdrant.NewValueFromFields(fields), nil
	case []any:
		vals := make([]*qdrant.Value, len(t))
		for i, val := range t {
			cv, err := anyToValue(val)
			if err != nil {
				return nil, err
			}
			vals[i] = cv
		}
		return qdrant.NewValueFromList(vals...), nil
	}
	return nil, fmt.Errorf("unsupported JSON payload type %T", v)
}

// payloadToJSON serializes a Qdrant payload map to a JSON string. An empty payload
// encodes to the empty string to keep rows compact.
func payloadToJSON(payload map[string]*qdrant.Value) (string, error) {
	if len(payload) == 0 {
		return "", nil
	}
	m := make(map[string]any, len(payload))
	for k, v := range payload {
		m[k] = valueToAny(v)
	}
	b, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("failed to encode payload: %w", err)
	}
	return string(b), nil
}

// jsonToPayload reverses payloadToJSON.
func jsonToPayload(s string) (map[string]*qdrant.Value, error) {
	if s == "" {
		return nil, nil
	}
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("failed to decode payload: %w", err)
	}
	out := make(map[string]*qdrant.Value, len(m))
	for k, v := range m {
		cv, err := anyToValue(v)
		if err != nil {
			return nil, err
		}
		out[k] = cv
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Vectors <-> JSON
// ---------------------------------------------------------------------------

// vectorOutputToJSON converts a retrieved vector into its JSON representation.
// Returns nil for absent or empty vectors.
func vectorOutputToJSON(v *qdrant.VectorOutput) *vectorJSON {
	if v == nil {
		return nil
	}
	if sparse := v.GetSparseVector(); sparse != nil {
		return &vectorJSON{Sparse: &sparseJSON{Indices: sparse.GetIndices(), Values: sparse.GetValues()}}
	}
	if multi := v.GetMultiVector(); multi != nil {
		rows := multi.GetVectors()
		data := make([][]float32, len(rows))
		for i, row := range rows {
			data[i] = row.GetData()
		}
		return &vectorJSON{Multi: data}
	}
	// Dense is the fallback, as any vector can be read as dense data.
	if dense := v.GetDenseVector(); dense != nil {
		data := dense.GetData()
		if len(data) == 0 {
			return nil
		}
		return &vectorJSON{Dense: data}
	}
	return nil
}

// retrievedVectorsToJSON serializes the vectors of a retrieved point to a JSON string.
// Returns the empty string when the point has no vectors.
func retrievedVectorsToJSON(p *qdrant.RetrievedPoint) (string, error) {
	if p.Vectors == nil {
		return "", nil
	}
	var out vectorsJSON
	if v := p.Vectors.GetVector(); v != nil {
		out.Single = vectorOutputToJSON(v)
	} else if vs := p.Vectors.GetVectors(); vs != nil {
		named := make(map[string]vectorJSON, len(vs.GetVectors()))
		for name, v := range vs.GetVectors() {
			if cv := vectorOutputToJSON(v); cv != nil {
				named[name] = *cv
			}
		}
		if len(named) > 0 {
			out.Named = named
		}
	}
	if out.Single == nil && out.Named == nil {
		return "", nil
	}
	b, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("failed to encode vectors: %w", err)
	}
	return string(b), nil
}

// vectorJSONToVector rebuilds a Qdrant upsert Vector from its JSON representation.
func vectorJSONToVector(v vectorJSON) *qdrant.Vector {
	switch {
	case v.Sparse != nil:
		return qdrant.NewVectorSparse(v.Sparse.Indices, v.Sparse.Values)
	case v.Multi != nil:
		return qdrant.NewVectorMulti(v.Multi)
	default:
		return qdrant.NewVectorDense(v.Dense)
	}
}

// jsonToVectors reverses retrievedVectorsToJSON, producing upsert-ready Vectors.
// Returns nil when the row has no vectors.
func jsonToVectors(s string) (*qdrant.Vectors, error) {
	if s == "" {
		return nil, nil
	}
	var in vectorsJSON
	if err := json.Unmarshal([]byte(s), &in); err != nil {
		return nil, fmt.Errorf("failed to decode vectors: %w", err)
	}
	if in.Single != nil {
		return &qdrant.Vectors{VectorsOptions: &qdrant.Vectors_Vector{Vector: vectorJSONToVector(*in.Single)}}, nil
	}
	if in.Named != nil {
		named := make(map[string]*qdrant.Vector, len(in.Named))
		for name, v := range in.Named {
			named[name] = vectorJSONToVector(v)
		}
		return qdrant.NewVectorsMap(named), nil
	}
	return nil, nil
}
