// document_test.go
package vector

import (
	"encoding/json"
	"sort"
	"strings"
	"testing"
)

// sortMetadataKeys sorts the keys of a metadata map.
func sortMetadataKeys(metadata map[string]interface{}) []string {
	var keys []string
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// createTestData creates test documents and results.
func createTestData() []Result {
	doc1 := &Document{
		ID: "doc1",
		Metadata: map[string]interface{}{
			"author": "Alice",
			"date":   "2023-04-01",
		},
		Segments: []*Segment{
			{Text: "Hello", Embedding: []float64{0.1, 0.2}},
			{Text: "World", Embedding: []float64{0.3, 0.4}},
		},
		Content: "Hello World",
	}

	doc2 := &Document{
		ID: "doc2",
		Metadata: map[string]interface{}{
			"author": "Bob",
			"date":   "2023-04-02",
		},
		Segments: []*Segment{
			{Text: "Foo", Embedding: []float64{0.5, 0.6}},
			{Text: "Bar", Embedding: []float64{0.7, 0.8}},
		},
		Content: "Foo Bar",
	}

	results := []Result{
		{Document: doc1, Segment: doc1.Segments[0], Similarity: 0.7},
		{Document: doc1, Segment: doc1.Segments[1], Similarity: 0.9},
		{Document: doc2, Segment: doc2.Segments[0], Similarity: 0.8},
		{Document: doc2, Segment: doc2.Segments[1], Similarity: 0.6},
	}

	return results
}

// MockFormatter is a mock implementation of the Formatter interface for testing.
type MockFormatter struct{}

func (mf MockFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	keys := sortMetadataKeys(metadata)

	var metadataStr string
	for _, key := range keys {
		metadataStr += key + ":" + metadata[key].(string) + " "
	}
	return docID + "|" + strings.TrimSpace(metadataStr) + "|" + content
}

// HTMLFormatter formats the document in HTML.
type HTMLFormatter struct{}

func (hf HTMLFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	keys := sortMetadataKeys(metadata)

	var metadataStr string
	for _, key := range keys {
		metadataStr += "<div>" + key + ": " + metadata[key].(string) + "</div>"
	}
	return "<div id='" + docID + "'>" + metadataStr + "<div>" + content + "</div></div>"
}

// JSONFormatter formats the document in JSON.
type JSONFormatter struct{}

func (jf JSONFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	// Define a struct to ensure the order of fields in JSON
	type JSONDocument struct {
		DocID    string                 `json:"docID"`
		Metadata map[string]interface{} `json:"metadata"`
		Content  string                 `json:"content"`
	}

	doc := JSONDocument{
		DocID:    docID,
		Metadata: metadata,
		Content:  content,
	}

	jsonData, err := json.Marshal(doc)
	if err != nil {
		return ""
	}
	return string(jsonData)
}

func TestAggregateResultsWithMockFormatter(t *testing.T) {
	results := createTestData()

	// Create a mock formatter.
	formatter := MockFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := "doc1|author:Alice date:2023-04-01|World\nHellodoc2|author:Bob date:2023-04-02|Foo\nBar"

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}

func TestAggregateResultsWithHTMLFormatter(t *testing.T) {
	results := createTestData()

	// Create an HTML formatter.
	formatter := HTMLFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := "<div id='doc1'><div>author: Alice</div><div>date: 2023-04-01</div><div>World\nHello</div></div>" +
		"<div id='doc2'><div>author: Bob</div><div>date: 2023-04-02</div><div>Foo\nBar</div></div>"

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}

func TestAggregateResultsWithJSONFormatter(t *testing.T) {
	results := createTestData()

	// Create a JSON formatter.
	formatter := JSONFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := `{"docID":"doc1","metadata":{"author":"Alice","date":"2023-04-01"},"content":"World\nHello"}{"docID":"doc2","metadata":{"author":"Bob","date":"2023-04-02"},"content":"Foo\nBar"}`

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}
