package vector

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"testing"
)

// MockEmbeddingFunc is a mock implementation of the EmbeddingFunc.
type MockEmbeddingFunc struct {
	mock.Mock
}

func (m *MockEmbeddingFunc) Embed(inputs []string, embeddingType string) ([][]float64, error) {
	args := m.Called(inputs, embeddingType)
	return args.Get(0).([][]float64), args.Error(1)
}

func TestNewCollection(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)

	tests := []struct {
		name                  string
		embeddingDocumentType string
		embeddingQueryType    string
		chunkSize             int
		chunkOverlap          int
		embeddingFunc         EmbeddingFunc
		wantErr               bool
	}{
		{"Valid Collection", "docType", "queryType", 100, 10, embeddingFunc.Embed, false},
		{"Missing Embedding Function", "docType", "queryType", 100, 10, nil, true},
		{"Invalid Chunk Size", "docType", "queryType", 0, 10, embeddingFunc.Embed, true},
		{"Invalid Chunk Overlap", "docType", "queryType", 100, -1, embeddingFunc.Embed, true},
		{"Overlap Greater Than Size", "docType", "queryType", 100, 101, embeddingFunc.Embed, true},
		{"Missing Collection Name", "", "queryType", 100, 10, embeddingFunc.Embed, true},
		{"Missing Document Type", "docType", "", 100, 10, embeddingFunc.Embed, true},
		{"Missing Query Type", "docType", "", 100, 10, embeddingFunc.Embed, true}, // 修正这里
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewCollection(tt.name, tt.embeddingDocumentType, tt.embeddingQueryType, tt.chunkSize, tt.chunkOverlap, tt.embeddingFunc)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollection_AddDocument(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	err := collection.AddDocument(doc)
	assert.NoError(t, err)

	// Test adding a document with the same ID
	err = collection.AddDocument(doc)
	assert.Error(t, err)

	// Test adding a document with empty ID
	err = collection.AddDocument(&Document{ID: "", Content: "Test content"})
	assert.Error(t, err)

	// Test adding a document with empty content
	err = collection.AddDocument(&Document{ID: "2", Content: ""})
	assert.Error(t, err)
}

func TestCollection_GetDocument(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	retrievedDoc, exists := collection.GetDocument("1")
	assert.True(t, exists)
	assert.Equal(t, doc, retrievedDoc)

	_, exists = collection.GetDocument("2")
	assert.False(t, exists)
}

func TestCollection_DeleteDocument(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	err := collection.DeleteDocument("1")
	assert.NoError(t, err)

	err = collection.DeleteDocument("1")
	assert.Error(t, err)
}

func TestCollection_UpdateDocument(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	updatedDoc := &Document{
		ID:      "1",
		Content: "Updated content.",
	}

	err := collection.UpdateDocument(updatedDoc)
	assert.NoError(t, err)

	retrievedDoc, _ := collection.GetDocument("1")
	assert.Equal(t, updatedDoc, retrievedDoc)

	// Test updating a non-existent document
	err = collection.UpdateDocument(&Document{ID: "2", Content: "Updated content"})
	assert.Error(t, err)

	// Test updating a document with empty ID
	err = collection.UpdateDocument(&Document{ID: "", Content: "Updated content"})
	assert.Error(t, err)

	// Test updating a document with empty content
	err = collection.UpdateDocument(&Document{ID: "1", Content: ""})
	assert.Error(t, err)
}

func TestCollection_Length(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	assert.Equal(t, 1, collection.Length())

	collection.AddDocument(&Document{ID: "2", Content: "Another document"})

	assert.Equal(t, 2, collection.Length())
}

func TestCollection_EmbedDocuments(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	err := collection.EmbedDocuments()
	assert.NoError(t, err)

	retrievedDoc, _ := collection.GetDocument("1")
	assert.NotEmpty(t, retrievedDoc.Segments)
}

func TestCollection_GetTopNSimilarDocuments(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	results, err := collection.GetTopNSimilarDocuments("test query", 1)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
}

func TestCollection_GetTopNSimilarDocumentsForQueries(t *testing.T) {
	embeddingFunc := new(MockEmbeddingFunc)
	embeddingFunc.On("Embed", mock.Anything, mock.Anything).Return([][]float64{{1.0, 2.0}}, nil)

	collection, _ := NewCollection("test", "docType", "queryType", 100, 10, embeddingFunc.Embed)

	doc := &Document{
		ID:      "1",
		Content: "This is a test document.",
	}

	collection.AddDocument(doc)

	queries := []string{"test query 1", "test query 2"}
	results, err := collection.GetTopNSimilarDocumentsForQueries(queries, 1)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
}
