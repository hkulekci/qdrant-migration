package cmd

import (
	"context"

	"github.com/qdrant/go-client/qdrant"
)

// qdrantGRPCClient wraps *qdrant.Client to satisfy commons.QdrantClient.
// Most methods are directly delegated to the underlying gRPC client.
type qdrantGRPCClient struct {
	*qdrant.Client
}

func (c *qdrantGRPCClient) ScrollWithOffset(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, *qdrant.PointId, error) {
	resp, err := c.Client.GetPointsClient().Scroll(ctx, req)
	if err != nil {
		return nil, nil, err
	}
	return resp.GetResult(), resp.GetNextPageOffset(), nil
}
