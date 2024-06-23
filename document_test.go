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
			"title":  "Hello World Title",
		},
		Segments: []*Segment{
			{Text: "Hello1", Embedding: []float64{0.1, 0.2}},
			{Text: "Not Wanted", Embedding: []float64{0.32, 0.42}},
			{Text: "World2", Embedding: []float64{0.3, 0.4}},
		},
		Content: "Hello1 World2",
	}

	doc2 := &Document{
		ID: "doc2",
		Metadata: map[string]interface{}{
			"author": "Bob",
			"title":  "Foo Bar Baz Title",
			"date":   "2023-04-02",
		},
		Segments: []*Segment{
			{Text: "Foo1", Embedding: []float64{0.5, 0.6}},
			{Text: "Bar2", Embedding: []float64{0.7, 0.8}},
			{Text: "Baz3", Embedding: []float64{0.91, 0.12}},
		},
		Content: "Foo1 Bar2 Baz3",
	}

	doc3 := &Document{
		ID: "doc3",
		Metadata: map[string]interface{}{
			"title":  "doc3 segment title",
			"author": "Charlie",
			"date":   "2023-04-03",
		},
		Segments: []*Segment{
			{Text: "doc3 segment", Embedding: []float64{0.9, 0.10}},
		},
		Content: "doc3 segment",
	}

	results := []Result{
		{Document: doc2, Segment: doc2.Segments[2], Similarity: 0.92},
		{Document: doc1, Segment: doc1.Segments[2], Similarity: 0.9},
		{Document: doc2, Segment: doc2.Segments[0], Similarity: 0.8},
		{Document: doc3, Segment: doc3.Segments[0], Similarity: 0.73},
		{Document: doc1, Segment: doc1.Segments[0], Similarity: 0.7},
		{Document: doc2, Segment: doc2.Segments[1], Similarity: 0.6},
	}

	return results
}

// MockFormatter is a mock implementation of the Formatter interface for testing.
type MockFormatter struct{}

func (mf *MockFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	keys := sortMetadataKeys(metadata)

	var metadataStr string
	for _, key := range keys {
		metadataStr += key + ":" + metadata[key].(string) + " "
	}
	return docID + "|" + strings.TrimSpace(metadataStr) + "|" + content
}

// HTMLFormatter formats the document in HTML.
type HTMLFormatter struct{}

func (hf *HTMLFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
	keys := sortMetadataKeys(metadata)

	var metadataStr string
	for _, key := range keys {
		metadataStr += "<div>" + key + ": " + metadata[key].(string) + "</div>"
	}
	return "<div id='" + docID + "'>" + metadataStr + "<div>" + content + "</div></div>"
}

// JSONFormatter formats the document in JSON.
type JSONFormatter struct{}

func (jf *JSONFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
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
	formatter := &MockFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := "doc2|author:Bob date:2023-04-02 title:Foo Bar Baz Title|Foo1\nBar2\nBaz3doc1|author:Alice date:2023-04-01 title:Hello World Title|Hello1\nWorld2doc3|author:Charlie date:2023-04-03 title:doc3 segment title|doc3 segment"

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}

func TestAggregateResultsWithHTMLFormatter(t *testing.T) {
	results := createTestData()

	// Create an HTML formatter.
	formatter := &HTMLFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := "<div id='doc2'><div>author: Bob</div><div>date: 2023-04-02</div><div>title: Foo Bar Baz Title</div><div>Foo1\nBar2\nBaz3</div></div>" +
		"<div id='doc1'><div>author: Alice</div><div>date: 2023-04-01</div><div>title: Hello World Title</div><div>Hello1\nWorld2</div></div>" +
		"<div id='doc3'><div>author: Charlie</div><div>date: 2023-04-03</div><div>title: doc3 segment title</div><div>doc3 segment</div></div>"

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}

func TestAggregateResultsWithJSONFormatter(t *testing.T) {
	results := createTestData()

	// Create a JSON formatter.
	formatter := &JSONFormatter{}

	// Aggregate the results.
	aggregated := AggregateResults(results, formatter)

	// Expected output.
	expected := `{"docID":"doc2","metadata":{"author":"Bob","date":"2023-04-02","title":"Foo Bar Baz Title"},"content":"Foo1\nBar2\nBaz3"}{"docID":"doc1","metadata":{"author":"Alice","date":"2023-04-01","title":"Hello World Title"},"content":"Hello1\nWorld2"}{"docID":"doc3","metadata":{"author":"Charlie","date":"2023-04-03","title":"doc3 segment title"},"content":"doc3 segment"}`

	// Check if the aggregated result matches the expected output.
	if aggregated != expected {
		t.Errorf("Expected:\n%s\nbut got:\n%s", expected, aggregated)
	}
}
