package vector

import (
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// EmbeddingFunc is a function that generates embeddings for a list of inputs.
type EmbeddingFunc func(inputs []string, embeddingType string) ([][]float64, error)

// Collection represents a collection of documents with metadata and an embedding function.
type Collection struct {
	Name                  string
	metadata              map[string]interface{}
	documents             map[string]*Document
	documentsLock         sync.RWMutex
	embeddingFunc         EmbeddingFunc
	embeddingDocumentType string // generate embeddings for documents.
	embeddingQueryType    string // generate embeddings for queries.
	ChunkSize             int
	ChunkOverlap          int
}

// NewCollection creates a new collection with the given name, split size, overlap size, and embedding function.
// SplitSize defines the size of each segment after splitting the document.
// OverlapSize defines the number of characters that will overlap between consecutive segments.
func NewCollection(name, embeddingDocumentType, embeddingQueryType string, chunkSize, chunkOverlap int, embeddingFunc EmbeddingFunc) (*Collection, error) {
	if embeddingFunc == nil {
		return nil, errors.New("embedding function is required")
	}
	if chunkSize <= 0 {
		return nil, errors.New("chunk size must be greater than zero")
	}
	if chunkOverlap < 0 {
		return nil, errors.New("chunk overlap must be greater than or equal to zero")
	}
	if chunkOverlap >= chunkSize {
		return nil, errors.New("chunk overlap must be less than chunk size")
	}
	if name == "" {
		return nil, errors.New("collection name is required")
	}
	if embeddingDocumentType == "" {
		return nil, errors.New("embedding document type is required")
	}
	if embeddingQueryType == "" {
		return nil, errors.New("embedding query type is required")
	}

	return &Collection{
		Name:                  name,
		metadata:              make(map[string]interface{}),
		documents:             make(map[string]*Document),
		embeddingFunc:         embeddingFunc,
		embeddingDocumentType: embeddingDocumentType,
		embeddingQueryType:    embeddingQueryType,
		ChunkSize:             chunkSize,
		ChunkOverlap:          chunkOverlap,
	}, nil
}

// AddDocument adds a document to the collection, generating embeddings if necessary.
func (c *Collection) AddDocument(doc *Document) error {
	c.documentsLock.Lock()
	defer c.documentsLock.Unlock()

	if doc.ID == "" {
		return errors.New("document ID is required")
	}

	if _, ok := c.documents[doc.ID]; ok {
		return fmt.Errorf("document with ID %s already exists", doc.ID)
	}

	if doc.Content == "" {
		return errors.New("document content is required")
	}

	// Split the content into segments.
	segments, err := c.splitText(doc.Content)
	if err != nil {
		return err
	}

	// Generate embeddings for each segment.
	embeddings, err := c.embeddingFunc(segments, c.embeddingDocumentType)
	if err != nil {
		return err
	}

	for i, embedding := range embeddings {
		// Normalize the embedding.
		norm, err := normalizeVector(embedding)
		if err != nil {
			return err
		}

		// Create a segment with the normalized embedding.
		doc.Segments = append(doc.Segments, &Segment{
			Text:      segments[i],
			Embedding: norm,
		})
	}

	// Add the document to the collection.
	c.documents[doc.ID] = doc
	return nil
}

// GetDocument retrieves a document from the collection by ID.
func (c *Collection) GetDocument(id string) (*Document, bool) {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	doc, ok := c.documents[id]
	return doc, ok
}

// GetDocuments retrieves all documents from the collection.
func (c *Collection) GetDocuments() map[string]Document {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	docs := make(map[string]Document, len(c.documents))
	for id, doc := range c.documents {
		docs[id] = *doc
	}
	return docs
}

// DeleteDocument removes a document from the collection by ID.
func (c *Collection) DeleteDocument(id string) error {
	c.documentsLock.Lock()
	defer c.documentsLock.Unlock()

	if _, ok := c.documents[id]; !ok {
		return fmt.Errorf("document with ID %s not found", id)
	}

	delete(c.documents, id)
	return nil
}

// UpdateDocument updates a document in the collection.
func (c *Collection) UpdateDocument(doc *Document) error {
	c.documentsLock.Lock()
	defer c.documentsLock.Unlock()

	if doc.ID == "" {
		return errors.New("document ID is required")
	}

	if _, ok := c.documents[doc.ID]; !ok {
		return fmt.Errorf("document with ID %s not found", doc.ID)
	}

	if doc.Content == "" {
		return errors.New("document content is required")
	}

	c.documents[doc.ID] = doc
	return nil
}

// Length returns the number of documents in the collection.
func (c *Collection) Length() int {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	return len(c.documents)
}

// EmbedDocuments splits the content of each document into segments and generates embeddings.
func (c *Collection) EmbedDocuments() error {
	c.documentsLock.Lock()
	defer c.documentsLock.Unlock()

	for _, doc := range c.documents {
		if doc.Content == "" {
			return errors.New("document content is required")
		}

		// Split the content into segments.
		segments, err := c.splitText(doc.Content)
		if err != nil {
			return err
		}

		// Generate embeddings for each segment.
		embeddings, err := c.embeddingFunc(segments, c.embeddingDocumentType)
		if err != nil {
			return err
		}

		// Normalize the embeddings.
		for i, embedding := range embeddings {
			norm, err := normalizeVector(embedding)
			if err != nil {
				return err
			}
			embeddings[i] = norm
		}

		// Create segments with embeddings.
		for i, segmentText := range segments {
			segment := Segment{
				Text:      segmentText,
				Embedding: embeddings[i],
			}
			doc.Segments = append(doc.Segments, &segment)
		}
	}

	return nil
}

// Result represents a single result from a query with a document and similarity score.
type Result struct {
	Document   *Document
	Segment    *Segment
	Similarity float64
}

// GetTopNSimilarDocuments retrieves the top N similar documents to the given query.
func (c *Collection) GetTopNSimilarDocuments(query string, topN int) ([]Result, error) {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	queryEmbedding, err := c.embeddingFunc([]string{query}, c.embeddingQueryType)
	if err != nil {
		return nil, err
	}

	if len(queryEmbedding) == 0 {
		return nil, errors.New("no embeddings generated for the query")
	}

	// Flatten the embeddings for all segments in the collection.
	var embeddings [][]float64
	var ids []string
	for docID, doc := range c.documents {
		for segmentIndex, segment := range doc.Segments {
			embeddings = append(embeddings, segment.Embedding)
			ids = append(ids, fmt.Sprintf("%s_%d", docID, segmentIndex))
		}
	}

	// Find the top N similar embeddings to the query embedding.
	similarities, err := getTopNSimilarEmbeddings(queryEmbedding[0], embeddings, ids, topN)
	if err != nil {
		return nil, err
	}

	// Helper function to parse the ID and segment index from the similarity ID.
	parseIDAndSegmentIndex := func(id string) (string, int, error) {
		lastUnderscoreIndex := strings.LastIndex(id, "_")
		if lastUnderscoreIndex == -1 {
			return "", 0, fmt.Errorf("invalid ID format: %s", id)
		}

		docID := id[:lastUnderscoreIndex]
		segmentIndex, err := strconv.Atoi(id[lastUnderscoreIndex+1:])
		if err != nil {
			return "", 0, err
		}

		return docID, segmentIndex, nil
	}

	var results []Result
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, sim := range similarities {
		wg.Add(1)
		go func(sim Similarity) {
			defer wg.Done()
			docID, segmentIndex, err := parseIDAndSegmentIndex(sim.ID)
			if err != nil {
				slog.Warn("failed to parse ID", "ID", sim.ID, "error", err)
				return // Skip invalid IDs.
			}

			doc, ok := c.documents[docID]
			if !ok {
				slog.Warn("document not found in collection", "docID", docID)
				return // Skip documents that are not found in the collection.
			}

			if segmentIndex < 0 || segmentIndex >= len(doc.Segments) {
				slog.Warn("segment index out of bounds for document", "segmentIndex", segmentIndex, "docID", docID)
				return // Skip segments that are out of bounds.
			}

			mu.Lock()
			results = append(results, Result{
				Document:   doc,
				Segment:    doc.Segments[segmentIndex],
				Similarity: sim.Score,
			})
			mu.Unlock()
		}(sim)
	}

	wg.Wait()

	// Sort the results by similarity score in descending order.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	return results, nil
}

// GetTopNSimilarDocumentsForQueries retrieves the top N similar documents for a list of queries.
func (c *Collection) GetTopNSimilarDocumentsForQueries(queries []string, topN int) ([]Result, error) {
	type queryResult struct {
		query   string
		results []Result
	}

	queryResults := make([]queryResult, len(queries))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errNum int

	// Process each query in parallel to get its top N similar documents.
	for i, query := range queries {
		wg.Add(1)
		go func(i int, query string) {
			defer wg.Done()
			results, queryErr := c.GetTopNSimilarDocuments(query, topN)
			if queryErr != nil {
				mu.Lock()
				slog.Warn("failed to get top N similar documents", "query", query, "error", queryErr)
				errNum++
				mu.Unlock()
				return
			}

			mu.Lock()
			queryResults[i] = queryResult{
				query:   query,
				results: results,
			}
			mu.Unlock()
		}(i, query)
	}

	wg.Wait()

	// Check if there was an error for all queries.
	if errNum == len(queries) {
		return nil, errors.New("failed to get top N similar documents for all queries")
	}

	// Use a map to track unique document IDs and collect the results.
	uniqueResults := make(map[string]Result)
	for _, qr := range queryResults {
		for _, res := range qr.results {
			docID := res.Document.ID
			// Use the document ID as the key to track unique results.
			if existingRes, exists := uniqueResults[docID]; !exists {
				uniqueResults[docID] = res
			} else {
				// If the document already exists, update the similarity score if the new score is higher.
				if res.Similarity > existingRes.Similarity {
					uniqueResults[docID] = res
				}
			}
		}
	}

	// Collect and sort the unique results by similarity score in descending order.
	var results []Result
	for _, res := range uniqueResults {
		results = append(results, res)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if len(results) > topN {
		results = results[:topN]
	}

	return results, nil
}

// splitText splits a long text into chunks of a maximum size with an overlap.
func (c *Collection) splitText(text string) ([]string, error) {
	var chunks []string
	runes := []rune(text)

	for i := 0; i < len(runes); i += c.ChunkSize - c.ChunkOverlap {
		end := i + c.ChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks, nil
}
