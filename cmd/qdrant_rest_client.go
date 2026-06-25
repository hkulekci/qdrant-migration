package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/qdrant/go-client/qdrant"
)

// qdrantRESTClient implements commons.QdrantClient using the Qdrant REST API.
type qdrantRESTClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func newQdrantRESTClient(baseURL, apiKey string) *qdrantRESTClient {
	return &qdrantRESTClient{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

func (c *qdrantRESTClient) doRequest(ctx context.Context, method, path string, body any) (json.RawMessage, error) {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("qdrant REST API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var envelope struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(respBody, &envelope); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return envelope.Result, nil
}

// --- Points ---

func (c *qdrantRESTClient) Upsert(ctx context.Context, req *qdrant.UpsertPoints) (*qdrant.UpdateResult, error) {
	waitParam := "?wait=false"
	if req.Wait != nil && *req.Wait {
		waitParam = "?wait=true"
	}

	type restPoint struct {
		ID      any                    `json:"id"`
		Payload map[string]any         `json:"payload,omitempty"`
		Vector  map[string]any         `json:"vector,omitempty"`
		Vectors map[string]any         `json:"vectors,omitempty"`
	}

	points := make([]restPoint, 0, len(req.Points))
	for _, p := range req.Points {
		rp := restPoint{
			ID: convertPointIDToREST(p.Id),
		}
		if p.Payload != nil {
			rp.Payload = convertPayloadToREST(p.Payload)
		}
		vecs := convertVectorsToREST(p.Vectors)
		if named, ok := vecs.(map[string]any); ok {
			rp.Vectors = named
		} else if vecs != nil {
			// Single unnamed vector
			rp.Vector = map[string]any{"": vecs}
		}
		points = append(points, rp)
	}

	body := map[string]any{
		"points": points,
	}

	if req.ShardKeySelector != nil {
		keys := make([]any, 0, len(req.ShardKeySelector.ShardKeys))
		for _, sk := range req.ShardKeySelector.ShardKeys {
			if kw := sk.GetKeyword(); kw != "" {
				keys = append(keys, kw)
			} else {
				keys = append(keys, sk.GetNumber())
			}
		}
		body["shard_key"] = keys
	}

	path := fmt.Sprintf("/collections/%s/points%s", req.CollectionName, waitParam)
	result, err := c.doRequest(ctx, http.MethodPut, path, body)
	if err != nil {
		return nil, err
	}

	var updateResult restUpdateResult
	if err := json.Unmarshal(result, &updateResult); err != nil {
		return nil, fmt.Errorf("failed to unmarshal update result: %w", err)
	}

	return updateResult.toProto(), nil
}

func (c *qdrantRESTClient) Count(ctx context.Context, req *qdrant.CountPoints) (uint64, error) {
	body := map[string]any{
		"exact": req.Exact != nil && *req.Exact,
	}
	if req.Filter != nil {
		body["filter"] = convertFilterToREST(req.Filter)
	}

	path := fmt.Sprintf("/collections/%s/points/count", req.CollectionName)
	result, err := c.doRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return 0, err
	}

	var countResult struct {
		Count uint64 `json:"count"`
	}
	if err := json.Unmarshal(result, &countResult); err != nil {
		return 0, fmt.Errorf("failed to unmarshal count result: %w", err)
	}

	return countResult.Count, nil
}

func (c *qdrantRESTClient) Get(ctx context.Context, req *qdrant.GetPoints) ([]*qdrant.RetrievedPoint, error) {
	ids := make([]any, 0, len(req.Ids))
	for _, id := range req.Ids {
		ids = append(ids, convertPointIDToREST(id))
	}

	body := map[string]any{
		"ids": ids,
	}
	if req.WithPayload != nil {
		body["with_payload"] = convertWithPayloadToREST(req.WithPayload)
	}
	if req.WithVectors != nil {
		body["with_vector"] = convertWithVectorsToREST(req.WithVectors)
	}

	path := fmt.Sprintf("/collections/%s/points", req.CollectionName)
	result, err := c.doRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}

	var restPoints []restRetrievedPoint
	if err := json.Unmarshal(result, &restPoints); err != nil {
		return nil, fmt.Errorf("failed to unmarshal points: %w", err)
	}

	points := make([]*qdrant.RetrievedPoint, 0, len(restPoints))
	for _, rp := range restPoints {
		points = append(points, rp.toProto())
	}
	return points, nil
}

func (c *qdrantRESTClient) Query(ctx context.Context, req *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
	body := map[string]any{}

	if req.CollectionName != "" {
		body["collection_name"] = req.CollectionName
	}
	if req.Query != nil {
		body["query"] = convertQueryToREST(req.Query)
	}
	if req.Limit != nil {
		body["limit"] = *req.Limit
	}
	if req.WithPayload != nil {
		body["with_payload"] = convertWithPayloadToREST(req.WithPayload)
	}
	if req.WithVectors != nil {
		body["with_vector"] = convertWithVectorsToREST(req.WithVectors)
	}
	if req.Filter != nil {
		body["filter"] = convertFilterToREST(req.Filter)
	}

	path := fmt.Sprintf("/collections/%s/points/query", req.CollectionName)
	result, err := c.doRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}

	var queryResult struct {
		Points []restScoredPoint `json:"points"`
	}
	if err := json.Unmarshal(result, &queryResult); err != nil {
		return nil, fmt.Errorf("failed to unmarshal query result: %w", err)
	}

	scored := make([]*qdrant.ScoredPoint, 0, len(queryResult.Points))
	for _, rp := range queryResult.Points {
		scored = append(scored, rp.toProto())
	}
	return scored, nil
}

func (c *qdrantRESTClient) Scroll(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, error) {
	points, _, err := c.scrollInternal(ctx, req)
	return points, err
}

func (c *qdrantRESTClient) ScrollWithOffset(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, *qdrant.PointId, error) {
	return c.scrollInternal(ctx, req)
}

func (c *qdrantRESTClient) scrollInternal(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, *qdrant.PointId, error) {
	body := map[string]any{}

	if req.Offset != nil {
		body["offset"] = convertPointIDToREST(req.Offset)
	}
	if req.Limit != nil {
		body["limit"] = *req.Limit
	}
	if req.WithPayload != nil {
		body["with_payload"] = convertWithPayloadToREST(req.WithPayload)
	}
	if req.WithVectors != nil {
		body["with_vector"] = convertWithVectorsToREST(req.WithVectors)
	}
	if req.Filter != nil {
		body["filter"] = convertFilterToREST(req.Filter)
	}

	path := fmt.Sprintf("/collections/%s/points/scroll", req.CollectionName)
	result, err := c.doRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, nil, err
	}

	var scrollResult struct {
		Points        []restRetrievedPoint `json:"points"`
		NextPageOffset json.RawMessage     `json:"next_page_offset"`
	}
	if err := json.Unmarshal(result, &scrollResult); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal scroll result: %w", err)
	}

	points := make([]*qdrant.RetrievedPoint, 0, len(scrollResult.Points))
	for _, rp := range scrollResult.Points {
		points = append(points, rp.toProto())
	}

	var nextOffset *qdrant.PointId
	if scrollResult.NextPageOffset != nil && string(scrollResult.NextPageOffset) != "null" {
		nextOffset = parsePointIDFromJSON(scrollResult.NextPageOffset)
	}

	return points, nextOffset, nil
}

// --- Collections ---

func (c *qdrantRESTClient) CreateFieldIndex(ctx context.Context, req *qdrant.CreateFieldIndexCollection) (*qdrant.UpdateResult, error) {
	waitParam := "?wait=false"
	if req.Wait != nil && *req.Wait {
		waitParam = "?wait=true"
	}

	body := map[string]any{
		"field_name": req.FieldName,
	}
	if req.FieldType != nil {
		body["field_schema"] = convertFieldTypeToREST(*req.FieldType)
	}

	path := fmt.Sprintf("/collections/%s/index%s", req.CollectionName, waitParam)
	result, err := c.doRequest(ctx, http.MethodPut, path, body)
	if err != nil {
		return nil, err
	}

	var updateResult restUpdateResult
	if err := json.Unmarshal(result, &updateResult); err != nil {
		return nil, fmt.Errorf("failed to unmarshal update result: %w", err)
	}

	return updateResult.toProto(), nil
}

func (c *qdrantRESTClient) CollectionExists(ctx context.Context, collectionName string) (bool, error) {
	path := fmt.Sprintf("/collections/%s/exists", collectionName)
	result, err := c.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return false, err
	}

	var existsResult struct {
		Exists bool `json:"exists"`
	}
	if err := json.Unmarshal(result, &existsResult); err != nil {
		return false, fmt.Errorf("failed to unmarshal exists result: %w", err)
	}

	return existsResult.Exists, nil
}

func (c *qdrantRESTClient) CreateCollection(ctx context.Context, req *qdrant.CreateCollection) error {
	body := convertCreateCollectionToREST(req)
	path := fmt.Sprintf("/collections/%s", req.CollectionName)
	_, err := c.doRequest(ctx, http.MethodPut, path, body)
	return err
}

func (c *qdrantRESTClient) DeleteCollection(ctx context.Context, collectionName string) error {
	path := fmt.Sprintf("/collections/%s", collectionName)
	_, err := c.doRequest(ctx, http.MethodDelete, path, nil)
	return err
}

func (c *qdrantRESTClient) GetCollectionInfo(ctx context.Context, collectionName string) (*qdrant.CollectionInfo, error) {
	path := fmt.Sprintf("/collections/%s", collectionName)
	result, err := c.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	var info restCollectionInfo
	if err := json.Unmarshal(result, &info); err != nil {
		return nil, fmt.Errorf("failed to unmarshal collection info: %w", err)
	}

	return info.toProto(), nil
}

func (c *qdrantRESTClient) CreateShardKey(ctx context.Context, collectionName string, req *qdrant.CreateShardKey) error {
	body := map[string]any{}
	if sk := req.ShardKey; sk != nil {
		if kw := sk.GetKeyword(); kw != "" {
			body["shard_key"] = kw
		} else {
			body["shard_key"] = sk.GetNumber()
		}
	}

	path := fmt.Sprintf("/collections/%s/shards", collectionName)
	_, err := c.doRequest(ctx, http.MethodPut, path, body)
	return err
}

func (c *qdrantRESTClient) Close() error {
	return nil
}
