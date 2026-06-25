package cmd

import (
	"encoding/json"
	"strconv"

	"github.com/qdrant/go-client/qdrant"
)

// --- REST response types ---

type restUpdateResult struct {
	OperationID uint64 `json:"operation_id"`
	Status      string `json:"status"`
}

func (r *restUpdateResult) toProto() *qdrant.UpdateResult {
	status := qdrant.UpdateStatus_Acknowledged
	if r.Status == "completed" {
		status = qdrant.UpdateStatus_Completed
	}
	return &qdrant.UpdateResult{
		OperationId: &r.OperationID,
		Status:      status,
	}
}

type restRetrievedPoint struct {
	ID       json.RawMessage        `json:"id"`
	Payload  map[string]any         `json:"payload,omitempty"`
	Vector   any                    `json:"vector,omitempty"`
	Vectors  any                    `json:"vectors,omitempty"`
	ShardKey any                    `json:"shard_key,omitempty"`
}

func (r *restRetrievedPoint) toProto() *qdrant.RetrievedPoint {
	point := &qdrant.RetrievedPoint{
		Id: parsePointIDFromJSON(r.ID),
	}

	if r.Payload != nil {
		point.Payload = convertRESTPayloadToProto(r.Payload)
	}

	point.Vectors = convertRESTVectorsToProto(r.Vector, r.Vectors)

	if r.ShardKey != nil {
		point.ShardKey = convertRESTShardKeyToProto(r.ShardKey)
	}

	return point
}

type restScoredPoint struct {
	ID       json.RawMessage        `json:"id"`
	Version  uint64                 `json:"version"`
	Score    float32                `json:"score"`
	Payload  map[string]any         `json:"payload,omitempty"`
	Vector   any                    `json:"vector,omitempty"`
	Vectors  any                    `json:"vectors,omitempty"`
	ShardKey any                    `json:"shard_key,omitempty"`
}

func (r *restScoredPoint) toProto() *qdrant.ScoredPoint {
	point := &qdrant.ScoredPoint{
		Id:      parsePointIDFromJSON(r.ID),
		Version: r.Version,
		Score:   r.Score,
	}

	if r.Payload != nil {
		point.Payload = convertRESTPayloadToProto(r.Payload)
	}

	point.Vectors = convertRESTVectorsToProto(r.Vector, r.Vectors)

	if r.ShardKey != nil {
		point.ShardKey = convertRESTShardKeyToProto(r.ShardKey)
	}

	return point
}

type restCollectionInfo struct {
	Status         string          `json:"status"`
	Config         json.RawMessage `json:"config"`
	PayloadSchema  map[string]any  `json:"payload_schema"`
	PointsCount    uint64          `json:"points_count"`
	VectorsCount   uint64          `json:"vectors_count"`
	SegmentsCount  uint64          `json:"segments_count"`
}

func (r *restCollectionInfo) toProto() *qdrant.CollectionInfo {
	info := &qdrant.CollectionInfo{
		PointsCount:   &r.PointsCount,
		SegmentsCount: r.SegmentsCount,
	}

	switch r.Status {
	case "green":
		info.Status = qdrant.CollectionStatus_Green
	case "yellow":
		info.Status = qdrant.CollectionStatus_Yellow
	case "red":
		info.Status = qdrant.CollectionStatus_Red
	case "grey":
		info.Status = qdrant.CollectionStatus_Grey
	}

	if r.Config != nil {
		var configMap map[string]json.RawMessage
		if err := json.Unmarshal(r.Config, &configMap); err == nil {
			info.Config = parseCollectionConfig(configMap)
		}
	}

	if r.PayloadSchema != nil {
		info.PayloadSchema = parsePayloadSchema(r.PayloadSchema)
	}

	return info
}

// --- Conversion helpers: Proto -> REST ---

func convertPointIDToREST(id *qdrant.PointId) any {
	if id == nil {
		return nil
	}
	if uuid := id.GetUuid(); uuid != "" {
		return uuid
	}
	return id.GetNum()
}

func convertPayloadToREST(payload map[string]*qdrant.Value) map[string]any {
	result := make(map[string]any, len(payload))
	for k, v := range payload {
		result[k] = convertValueToREST(v)
	}
	return result
}

func convertValueToREST(v *qdrant.Value) any {
	if v == nil {
		return nil
	}
	switch val := v.GetKind().(type) {
	case *qdrant.Value_NullValue:
		return nil
	case *qdrant.Value_BoolValue:
		return val.BoolValue
	case *qdrant.Value_IntegerValue:
		return val.IntegerValue
	case *qdrant.Value_DoubleValue:
		return val.DoubleValue
	case *qdrant.Value_StringValue:
		return val.StringValue
	case *qdrant.Value_ListValue:
		items := val.ListValue.GetValues()
		result := make([]any, len(items))
		for i, item := range items {
			result[i] = convertValueToREST(item)
		}
		return result
	case *qdrant.Value_StructValue:
		fields := val.StructValue.GetFields()
		result := make(map[string]any, len(fields))
		for k, v := range fields {
			result[k] = convertValueToREST(v)
		}
		return result
	default:
		return nil
	}
}

func convertVectorsToREST(vectors *qdrant.Vectors) any {
	if vectors == nil {
		return nil
	}
	switch v := vectors.VectorsOptions.(type) {
	case *qdrant.Vectors_Vector:
		return convertSingleVectorToREST(v.Vector)
	case *qdrant.Vectors_Vectors:
		result := make(map[string]any, len(v.Vectors.Vectors))
		for name, vec := range v.Vectors.Vectors {
			result[name] = convertSingleVectorToREST(vec)
		}
		return result
	}
	return nil
}

func convertSingleVectorToREST(v *qdrant.Vector) any {
	if v == nil {
		return nil
	}
	switch vec := v.Vector.(type) {
	case *qdrant.Vector_Dense:
		return vec.Dense.Data
	case *qdrant.Vector_Sparse:
		return map[string]any{
			"indices": vec.Sparse.Indices,
			"values":  vec.Sparse.Values,
		}
	case *qdrant.Vector_MultiDense:
		return vec.MultiDense
	}
	return nil
}

func convertWithPayloadToREST(wp *qdrant.WithPayloadSelector) any {
	if wp == nil {
		return true
	}
	switch sel := wp.SelectorOptions.(type) {
	case *qdrant.WithPayloadSelector_Enable:
		return sel.Enable
	case *qdrant.WithPayloadSelector_Include:
		return sel.Include.Fields
	case *qdrant.WithPayloadSelector_Exclude:
		return map[string]any{"exclude": sel.Exclude.Fields}
	}
	return true
}

func convertWithVectorsToREST(wv *qdrant.WithVectorsSelector) any {
	if wv == nil {
		return false
	}
	switch sel := wv.SelectorOptions.(type) {
	case *qdrant.WithVectorsSelector_Enable:
		return sel.Enable
	case *qdrant.WithVectorsSelector_Include:
		return sel.Include.Names
	}
	return false
}

func convertFilterToREST(filter *qdrant.Filter) map[string]any {
	if filter == nil {
		return nil
	}
	// Pass through as a generic map; for the migration tool the filter
	// is typically nil, so a minimal implementation is sufficient.
	result := make(map[string]any)
	if len(filter.Must) > 0 {
		result["must"] = convertConditionsToREST(filter.Must)
	}
	if len(filter.MustNot) > 0 {
		result["must_not"] = convertConditionsToREST(filter.MustNot)
	}
	if len(filter.Should) > 0 {
		result["should"] = convertConditionsToREST(filter.Should)
	}
	return result
}

func convertConditionsToREST(conditions []*qdrant.Condition) []any {
	result := make([]any, 0, len(conditions))
	for _, cond := range conditions {
		// Minimal implementation: serialize conditions as generic maps
		data, err := json.Marshal(cond)
		if err == nil {
			var m any
			if json.Unmarshal(data, &m) == nil {
				result = append(result, m)
			}
		}
	}
	return result
}

func convertQueryToREST(q *qdrant.Query) any {
	if q == nil {
		return nil
	}
	switch v := q.Variant.(type) {
	case *qdrant.Query_Sample:
		switch v.Sample {
		case qdrant.Sample_Random:
			return map[string]any{"sample": "random"}
		}
	}
	return nil
}

func convertFieldTypeToREST(ft qdrant.FieldType) string {
	switch ft {
	case qdrant.FieldType_FieldTypeKeyword:
		return "keyword"
	case qdrant.FieldType_FieldTypeInteger:
		return "integer"
	case qdrant.FieldType_FieldTypeFloat:
		return "float"
	case qdrant.FieldType_FieldTypeGeo:
		return "geo"
	case qdrant.FieldType_FieldTypeText:
		return "text"
	case qdrant.FieldType_FieldTypeBool:
		return "bool"
	case qdrant.FieldType_FieldTypeDatetime:
		return "datetime"
	case qdrant.FieldType_FieldTypeUuid:
		return "uuid"
	}
	return "keyword"
}

func convertCreateCollectionToREST(req *qdrant.CreateCollection) map[string]any {
	body := make(map[string]any)

	if req.VectorsConfig != nil {
		body["vectors"] = convertVectorsConfigToREST(req.VectorsConfig)
	}
	if req.SparseVectorsConfig != nil {
		sparse := make(map[string]any)
		for name, params := range req.SparseVectorsConfig.Map {
			sparseParams := make(map[string]any)
			if params != nil {
				if params.Index != nil {
					sparseParams["index"] = map[string]any{
						"full_scan_threshold": params.Index.FullScanThreshold,
					}
				}
			}
			sparse[name] = sparseParams
		}
		body["sparse_vectors"] = sparse
	}
	if req.ShardNumber != nil {
		body["shard_number"] = *req.ShardNumber
	}
	if req.OnDiskPayload != nil {
		body["on_disk_payload"] = *req.OnDiskPayload
	}
	if req.ReplicationFactor != nil {
		body["replication_factor"] = *req.ReplicationFactor
	}
	if req.WriteConsistencyFactor != nil {
		body["write_consistency_factor"] = *req.WriteConsistencyFactor
	}
	if req.HnswConfig != nil {
		body["hnsw_config"] = convertHnswConfigToREST(req.HnswConfig)
	}
	if req.WalConfig != nil {
		body["wal_config"] = convertWalConfigToREST(req.WalConfig)
	}
	if req.OptimizersConfig != nil {
		body["optimizers_config"] = convertOptimizersConfigToREST(req.OptimizersConfig)
	}
	if req.QuantizationConfig != nil {
		body["quantization_config"] = convertQuantizationConfigToREST(req.QuantizationConfig)
	}

	return body
}

func convertVectorsConfigToREST(vc *qdrant.VectorsConfig) any {
	switch v := vc.Config.(type) {
	case *qdrant.VectorsConfig_Params:
		return convertVectorParamsToREST(v.Params)
	case *qdrant.VectorsConfig_ParamsMap:
		result := make(map[string]any, len(v.ParamsMap.Map))
		for name, params := range v.ParamsMap.Map {
			result[name] = convertVectorParamsToREST(params)
		}
		return result
	}
	return nil
}

func convertVectorParamsToREST(p *qdrant.VectorParams) map[string]any {
	result := map[string]any{
		"size":     p.Size,
		"distance": convertDistanceToREST(p.Distance),
	}
	if p.OnDisk != nil {
		result["on_disk"] = *p.OnDisk
	}
	return result
}

func convertDistanceToREST(d qdrant.Distance) string {
	switch d {
	case qdrant.Distance_Cosine:
		return "Cosine"
	case qdrant.Distance_Euclid:
		return "Euclid"
	case qdrant.Distance_Dot:
		return "Dot"
	case qdrant.Distance_Manhattan:
		return "Manhattan"
	}
	return "Cosine"
}

func convertHnswConfigToREST(h *qdrant.HnswConfigDiff) map[string]any {
	result := make(map[string]any)
	if h.M != nil {
		result["m"] = *h.M
	}
	if h.EfConstruct != nil {
		result["ef_construct"] = *h.EfConstruct
	}
	if h.FullScanThreshold != nil {
		result["full_scan_threshold"] = *h.FullScanThreshold
	}
	if h.MaxIndexingThreads != nil {
		result["max_indexing_threads"] = *h.MaxIndexingThreads
	}
	if h.OnDisk != nil {
		result["on_disk"] = *h.OnDisk
	}
	return result
}

func convertWalConfigToREST(w *qdrant.WalConfigDiff) map[string]any {
	result := make(map[string]any)
	if w.WalCapacityMb != nil {
		result["wal_capacity_mb"] = *w.WalCapacityMb
	}
	if w.WalSegmentsAhead != nil {
		result["wal_segments_ahead"] = *w.WalSegmentsAhead
	}
	return result
}

func convertOptimizersConfigToREST(o *qdrant.OptimizersConfigDiff) map[string]any {
	result := make(map[string]any)
	if o.IndexingThreshold != nil {
		result["indexing_threshold"] = *o.IndexingThreshold
	}
	if o.DefaultSegmentNumber != nil {
		result["default_segment_number"] = *o.DefaultSegmentNumber
	}
	if o.MaxSegmentSize != nil {
		result["max_segment_size"] = *o.MaxSegmentSize
	}
	if o.MemmapThreshold != nil {
		result["memmap_threshold"] = *o.MemmapThreshold
	}
	if o.FlushIntervalSec != nil {
		result["flush_interval_sec"] = *o.FlushIntervalSec
	}
	if o.MaxOptimizationThreads != nil {
		result["max_optimization_threads"] = *o.MaxOptimizationThreads
	}
	return result
}

func convertQuantizationConfigToREST(q *qdrant.QuantizationConfig) any {
	if q == nil {
		return nil
	}
	switch v := q.Quantization.(type) {
	case *qdrant.QuantizationConfig_Scalar:
		result := map[string]any{
			"scalar": map[string]any{
				"type": convertScalarQuantizationType(v.Scalar.Type),
			},
		}
		if v.Scalar.Quantile != nil {
			result["scalar"].(map[string]any)["quantile"] = *v.Scalar.Quantile
		}
		if v.Scalar.AlwaysRam != nil {
			result["scalar"].(map[string]any)["always_ram"] = *v.Scalar.AlwaysRam
		}
		return result
	case *qdrant.QuantizationConfig_Product:
		result := map[string]any{
			"product": map[string]any{
				"compression": convertCompressionRatio(v.Product.Compression),
			},
		}
		if v.Product.AlwaysRam != nil {
			result["product"].(map[string]any)["always_ram"] = *v.Product.AlwaysRam
		}
		return result
	case *qdrant.QuantizationConfig_Binary:
		result := map[string]any{
			"binary": map[string]any{},
		}
		if v.Binary.AlwaysRam != nil {
			result["binary"].(map[string]any)["always_ram"] = *v.Binary.AlwaysRam
		}
		return result
	}
	return nil
}

func convertScalarQuantizationType(t qdrant.QuantizationType) string {
	switch t {
	case qdrant.QuantizationType_Int8:
		return "int8"
	}
	return "int8"
}

func convertCompressionRatio(c qdrant.CompressionRatio) string {
	switch c {
	case qdrant.CompressionRatio_x4:
		return "x4"
	case qdrant.CompressionRatio_x8:
		return "x8"
	case qdrant.CompressionRatio_x16:
		return "x16"
	case qdrant.CompressionRatio_x32:
		return "x32"
	case qdrant.CompressionRatio_x64:
		return "x64"
	}
	return "x4"
}

// --- Conversion helpers: REST -> Proto ---

func parsePointIDFromJSON(raw json.RawMessage) *qdrant.PointId {
	if raw == nil || string(raw) == "null" {
		return nil
	}

	// Try as number first
	var num uint64
	if err := json.Unmarshal(raw, &num); err == nil {
		return qdrant.NewIDNum(num)
	}

	// Try as string (UUID)
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		// Check if it's a numeric string
		if n, err := strconv.ParseUint(str, 10, 64); err == nil {
			return qdrant.NewIDNum(n)
		}
		return qdrant.NewIDUUID(str)
	}

	return nil
}

func convertRESTPayloadToProto(payload map[string]any) map[string]*qdrant.Value {
	return qdrant.NewValueMap(payload)
}

func convertRESTVectorsToProto(vector any, vectors any) *qdrant.VectorsOutput {
	if vectors != nil {
		switch v := vectors.(type) {
		case map[string]any:
			named := make(map[string]*qdrant.VectorOutput, len(v))
			for name, vec := range v {
				named[name] = parseVectorOutput(vec)
			}
			return &qdrant.VectorsOutput{
				VectorsOptions: &qdrant.VectorsOutput_Vectors{
					Vectors: &qdrant.NamedVectorsOutput{Vectors: named},
				},
			}
		}
	}

	if vector != nil {
		vo := parseVectorOutput(vector)
		if vo != nil {
			return &qdrant.VectorsOutput{
				VectorsOptions: &qdrant.VectorsOutput_Vector{Vector: vo},
			}
		}
	}

	return nil
}

func parseVectorOutput(v any) *qdrant.VectorOutput {
	switch vec := v.(type) {
	case []any:
		data := make([]float32, len(vec))
		for i, val := range vec {
			switch n := val.(type) {
			case float64:
				data[i] = float32(n)
			case float32:
				data[i] = n
			}
		}
		return &qdrant.VectorOutput{
			Vector: &qdrant.VectorOutput_Dense{
				Dense: &qdrant.DenseVector{Data: data},
			},
		}
	case map[string]any:
		// Could be a sparse vector with "indices" and "values"
		if indices, ok := vec["indices"]; ok {
			if values, ok := vec["values"]; ok {
				return parseSparseVectorOutput(indices, values)
			}
		}
	}
	return nil
}

func parseSparseVectorOutput(indicesRaw, valuesRaw any) *qdrant.VectorOutput {
	var indices []uint32
	var values []float32

	switch idx := indicesRaw.(type) {
	case []any:
		indices = make([]uint32, len(idx))
		for i, v := range idx {
			switch n := v.(type) {
			case float64:
				indices[i] = uint32(n)
			}
		}
	}

	switch vals := valuesRaw.(type) {
	case []any:
		values = make([]float32, len(vals))
		for i, v := range vals {
			switch n := v.(type) {
			case float64:
				values[i] = float32(n)
			}
		}
	}

	if len(indices) > 0 && len(values) > 0 {
		return &qdrant.VectorOutput{
			Vector: &qdrant.VectorOutput_Sparse{
				Sparse: &qdrant.SparseVector{
					Indices: indices,
					Values:  values,
				},
			},
		}
	}
	return nil
}

func convertRESTShardKeyToProto(sk any) *qdrant.ShardKey {
	switch v := sk.(type) {
	case string:
		return &qdrant.ShardKey{Key: &qdrant.ShardKey_Keyword{Keyword: v}}
	case float64:
		return &qdrant.ShardKey{Key: &qdrant.ShardKey_Number{Number: uint64(v)}}
	}
	return nil
}

// --- Collection config parsing (REST -> Proto) ---

func parseCollectionConfig(configMap map[string]json.RawMessage) *qdrant.CollectionConfig {
	config := &qdrant.CollectionConfig{}

	if paramsRaw, ok := configMap["params"]; ok {
		config.Params = parseCollectionParams(paramsRaw)
	}
	if hnswRaw, ok := configMap["hnsw_config"]; ok {
		config.HnswConfig = parseHnswConfig(hnswRaw)
	}
	if walRaw, ok := configMap["wal_config"]; ok {
		config.WalConfig = parseWalConfig(walRaw)
	}
	if optRaw, ok := configMap["optimizer_config"]; ok {
		config.OptimizerConfig = parseOptimizerConfig(optRaw)
	}
	if quantRaw, ok := configMap["quantization_config"]; ok {
		config.QuantizationConfig = parseQuantizationConfig(quantRaw)
	}

	return config
}

func parseCollectionParams(raw json.RawMessage) *qdrant.CollectionParams {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	params := &qdrant.CollectionParams{}

	if v, ok := m["shard_number"]; ok {
		var n uint32
		if json.Unmarshal(v, &n) == nil {
			params.ShardNumber = n
		}
	}
	if v, ok := m["on_disk_payload"]; ok {
		var b bool
		if json.Unmarshal(v, &b) == nil {
			params.OnDiskPayload = b
		}
	}
	if v, ok := m["replication_factor"]; ok {
		var n uint32
		if json.Unmarshal(v, &n) == nil {
			params.ReplicationFactor = &n
		}
	}
	if v, ok := m["write_consistency_factor"]; ok {
		var n uint32
		if json.Unmarshal(v, &n) == nil {
			params.WriteConsistencyFactor = &n
		}
	}
	if v, ok := m["vectors"]; ok {
		params.VectorsConfig = parseVectorsConfig(v)
	}
	if v, ok := m["sparse_vectors"]; ok {
		params.SparseVectorsConfig = parseSparseVectorsConfig(v)
	}

	return params
}

func parseVectorsConfig(raw json.RawMessage) *qdrant.VectorsConfig {
	// Try as a single vector config first
	var single struct {
		Size     uint64 `json:"size"`
		Distance string `json:"distance"`
	}
	if err := json.Unmarshal(raw, &single); err == nil && single.Size > 0 {
		return &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     single.Size,
					Distance: parseDistance(single.Distance),
				},
			},
		}
	}

	// Try as named vector map
	var namedMap map[string]json.RawMessage
	if err := json.Unmarshal(raw, &namedMap); err == nil {
		paramsMap := make(map[string]*qdrant.VectorParams)
		for name, paramRaw := range namedMap {
			var p struct {
				Size     uint64 `json:"size"`
				Distance string `json:"distance"`
			}
			if json.Unmarshal(paramRaw, &p) == nil && p.Size > 0 {
				paramsMap[name] = &qdrant.VectorParams{
					Size:     p.Size,
					Distance: parseDistance(p.Distance),
				}
			}
		}
		if len(paramsMap) > 0 {
			return &qdrant.VectorsConfig{
				Config: &qdrant.VectorsConfig_ParamsMap{
					ParamsMap: &qdrant.VectorParamsMap{Map: paramsMap},
				},
			}
		}
	}

	return nil
}

func parseSparseVectorsConfig(raw json.RawMessage) *qdrant.SparseVectorConfig {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	result := &qdrant.SparseVectorConfig{
		Map: make(map[string]*qdrant.SparseVectorParams),
	}
	for name := range m {
		result.Map[name] = &qdrant.SparseVectorParams{}
	}
	return result
}

func parseDistance(s string) qdrant.Distance {
	switch s {
	case "Cosine":
		return qdrant.Distance_Cosine
	case "Euclid":
		return qdrant.Distance_Euclid
	case "Dot":
		return qdrant.Distance_Dot
	case "Manhattan":
		return qdrant.Distance_Manhattan
	}
	return qdrant.Distance_Cosine
}

func parseHnswConfig(raw json.RawMessage) *qdrant.HnswConfigDiff {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	config := &qdrant.HnswConfigDiff{}
	if v, ok := m["m"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.M = &n
		}
	}
	if v, ok := m["ef_construct"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.EfConstruct = &n
		}
	}
	if v, ok := m["full_scan_threshold"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.FullScanThreshold = &n
		}
	}
	if v, ok := m["max_indexing_threads"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.MaxIndexingThreads = &n
		}
	}
	if v, ok := m["on_disk"]; ok {
		var b bool
		if json.Unmarshal(v, &b) == nil {
			config.OnDisk = &b
		}
	}
	return config
}

func parseWalConfig(raw json.RawMessage) *qdrant.WalConfigDiff {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	config := &qdrant.WalConfigDiff{}
	if v, ok := m["wal_capacity_mb"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.WalCapacityMb = &n
		}
	}
	if v, ok := m["wal_segments_ahead"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.WalSegmentsAhead = &n
		}
	}
	return config
}

func parseOptimizerConfig(raw json.RawMessage) *qdrant.OptimizersConfigDiff {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	config := &qdrant.OptimizersConfigDiff{}
	if v, ok := m["indexing_threshold"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.IndexingThreshold = &n
		}
	}
	if v, ok := m["default_segment_number"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.DefaultSegmentNumber = &n
		}
	}
	if v, ok := m["max_segment_size"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.MaxSegmentSize = &n
		}
	}
	if v, ok := m["memmap_threshold"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.MemmapThreshold = &n
		}
	}
	if v, ok := m["flush_interval_sec"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.FlushIntervalSec = &n
		}
	}
	if v, ok := m["max_optimization_threads"]; ok {
		var n uint64
		if json.Unmarshal(v, &n) == nil {
			config.MaxOptimizationThreads = &qdrant.MaxOptimizationThreads{
				Variant: &qdrant.MaxOptimizationThreads_Value{Value: n},
			}
		}
	}
	return config
}

func parseQuantizationConfig(raw json.RawMessage) *qdrant.QuantizationConfig {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	if scalarRaw, ok := m["scalar"]; ok {
		var scalar struct {
			Type     string   `json:"type"`
			Quantile *float32 `json:"quantile"`
			AlwaysRam *bool   `json:"always_ram"`
		}
		if json.Unmarshal(scalarRaw, &scalar) == nil {
			sq := &qdrant.ScalarQuantization{
				Type: qdrant.QuantizationType_Int8,
			}
			sq.Quantile = scalar.Quantile
			sq.AlwaysRam = scalar.AlwaysRam
			return &qdrant.QuantizationConfig{
				Quantization: &qdrant.QuantizationConfig_Scalar{Scalar: sq},
			}
		}
	}

	if binaryRaw, ok := m["binary"]; ok {
		var binary struct {
			AlwaysRam *bool `json:"always_ram"`
		}
		if json.Unmarshal(binaryRaw, &binary) == nil {
			bq := &qdrant.BinaryQuantization{}
			bq.AlwaysRam = binary.AlwaysRam
			return &qdrant.QuantizationConfig{
				Quantization: &qdrant.QuantizationConfig_Binary{Binary: bq},
			}
		}
	}

	return nil
}

func parsePayloadSchema(schema map[string]any) map[string]*qdrant.PayloadSchemaInfo {
	result := make(map[string]*qdrant.PayloadSchemaInfo)
	for name, info := range schema {
		infoMap, ok := info.(map[string]any)
		if !ok {
			continue
		}

		schemaInfo := &qdrant.PayloadSchemaInfo{}
		if dataType, ok := infoMap["data_type"].(string); ok {
			schemaInfo.DataType = parsePayloadSchemaType(dataType)
		}

		result[name] = schemaInfo
	}
	return result
}

func parsePayloadSchemaType(s string) qdrant.PayloadSchemaType {
	switch s {
	case "keyword":
		return qdrant.PayloadSchemaType_Keyword
	case "integer":
		return qdrant.PayloadSchemaType_Integer
	case "float":
		return qdrant.PayloadSchemaType_Float
	case "geo":
		return qdrant.PayloadSchemaType_Geo
	case "text":
		return qdrant.PayloadSchemaType_Text
	case "bool":
		return qdrant.PayloadSchemaType_Bool
	case "datetime":
		return qdrant.PayloadSchemaType_Datetime
	case "uuid":
		return qdrant.PayloadSchemaType_Uuid
	}
	return qdrant.PayloadSchemaType_Keyword
}
