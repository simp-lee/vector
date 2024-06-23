# Vector-Based Document Collection Library

This repository contains a Go implementation for managing and querying a collection of documents using vector embeddings. The core functionality includes adding, updating, deleting documents, and retrieving the most similar documents based on query embeddings. This library is particularly useful for applications involving natural language processing, information retrieval, and semantic search.

## Features

- **Document Management**: Add, update, retrieve, and delete documents with ease.
- **Embedding Generation**: Generate embeddings for documents and queries using a customizable embedding function.
- **Segmentation**: Split documents into manageable segments with optional overlap.
- **Similarity Search**: Retrieve the top N most similar documents to a given query based on cosine similarity of vector embeddings.
- **Concurrent Processing**: Utilize Go's concurrency features to handle multiple queries and document processing efficiently.

## Installation

To install the Vector Library, use the following command:

```shell
go get github.com/simp-lee/vector
```

## Usage

### Creating a Collection

To create a new collection, use the `NewCollection` function:

```go
package main

import (
	"github.com/simp-lee/vector"
	"log"
)

func main() {
	embeddingFunc := func(inputs []string, embeddingType string) ([][]float64, error) {
		// Implement your embedding generation logic here
		return [][]float64{}, nil
	}

	collection, err := vector.NewCollection("MyCollection", "document", "query", 100, 10, embeddingFunc)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Use the collection to manage documents and perform searches
}
```

### Adding a Document

To add a document to the collection, use the `AddDocument` function:

```go
doc := &vector.Document{
	ID:       "doc1",
	Metadata: map[string]interface{}{"author": "John Doe"},
	Content:  "This is a sample document content.",
}

err = collection.AddDocument(doc)
if err != nil {
	log.Fatalf("Failed to add document: %v", err)
}
```

### Retrieving Documents

**Retrieving a Document by ID**

```go
doc, found := collection.GetDocument("doc1")
if !found {
	log.Println("Document not found")
} else {
	log.Printf("Document found: %+v", doc)
}

// Use the document's metadata and content fields
```

**Retrieving Top N Similar Documents for a Query**

```go
results, err := collection.GetTopNSimilarDocuments("sample query", 5)
if err != nil {
	log.Fatalf("Failed to get top N similar documents: %v", err)
}

for _, result := range results {
	log.Printf("Document ID: %s, Similarity: %f\n", result.Document.ID, result.Similarity)
}
```

**Retrieving Top N Similar Documents for Multiple Queries**

```go
queries := []string{"query1", "query2", "query3"}
results, err := collection.GetTopNSimilarDocumentsForQueries(queries, 5)
if err != nil {
	log.Fatalf("Failed to get top N similar documents for queries: %v", err)
}

for _, result := range results {
	log.Printf("Document ID: %s, Similarity: %f\n", result.Document.ID, result.Similarity)
}
```

### Aggregating Results

To aggregate the results of multiple queries into a formatted string:

```go
type SimpleFormatter struct{}

func (f *SimpleFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	return fmt.Sprintf("Document ID: %s\nMetadata: %v\nContent: %s\n", docID, metadata, content)
}

queries := []string{"query one", "query two"}
results, err := collection.GetTopNSimilarDocumentsForQueries(queries, 5)
if err != nil {
	log.Fatalf("Failed to retrieve similar documents for queries: %v", err)
}

formatter := &SimpleFormatter{}
aggregatedResults := vector.AggregateResults(results, formatter)
fmt.Println(aggregatedResults)
```

### Updating Documents

To update a document in the collection, use the `UpdateDocument` function:

```go
doc := &vector.Document{
	ID:       "doc1",
	Metadata: map[string]interface{}{"author": "Jane Smith"},
	Content:  "This is an updated document content.",
}

err = collection.UpdateDocument(doc)
if err != nil {
	log.Fatalf("Failed to update document: %v", err)
}
```

### Deleting Documents

To delete a document from the collection, use the `DeleteDocument` function:

```go
err = collection.DeleteDocument("doc1")
if err != nil {
	log.Fatalf("Failed to delete document: %v", err)
}
```

### Generating Embeddings

To generate embeddings for a document or query, use the `GenerateEmbeddings` function:

```go
doc := &vector.Document{
	ID:       "doc1",
	Metadata: map[string]interface{}{"author": "John Doe"},
	Content:  "This is a sample document content.",
}

embeddings, err := collection.GenerateEmbeddings(doc)
if err != nil {
	log.Fatalf("Failed to generate embeddings: %v", err)
}

// Use the embeddings for similarity search
```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or enhancements.

## License

This project is licensed under the MIT License.
