package commons

import (
	"context"

	"github.com/qdrant/go-client/qdrant"
)

// QdrantClient defines the interface for interacting with a Qdrant server.
// Both gRPC and REST implementations satisfy this interface.
type QdrantClient interface {
	Upsert(ctx context.Context, req *qdrant.UpsertPoints) (*qdrant.UpdateResult, error)
	Count(ctx context.Context, req *qdrant.CountPoints) (uint64, error)
	Get(ctx context.Context, req *qdrant.GetPoints) ([]*qdrant.RetrievedPoint, error)
	Query(ctx context.Context, req *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error)
	Scroll(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, error)
	ScrollWithOffset(ctx context.Context, req *qdrant.ScrollPoints) ([]*qdrant.RetrievedPoint, *qdrant.PointId, error)
	CreateFieldIndex(ctx context.Context, req *qdrant.CreateFieldIndexCollection) (*qdrant.UpdateResult, error)
	CollectionExists(ctx context.Context, collectionName string) (bool, error)
	CreateCollection(ctx context.Context, req *qdrant.CreateCollection) error
	DeleteCollection(ctx context.Context, collectionName string) error
	GetCollectionInfo(ctx context.Context, collectionName string) (*qdrant.CollectionInfo, error)
	CreateShardKey(ctx context.Context, collectionName string, req *qdrant.CreateShardKey) error
	Close() error
}
