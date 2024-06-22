package vector

import (
	"sort"
	"strings"
)

// Segment represents a segment of a document that has been split into smaller pieces.
// Each segment has a corresponding embedding.
type Segment struct {
	Text      string
	Embedding []float64
}

// Document represents a document with metadata, segments, and content.
type Document struct {
	ID       string
	Metadata map[string]interface{}
	Segments []*Segment
	Content  string
}

// Formatter formats a document for display or export to a file format
// such as CSV or JSON etc. based on the given formatter interface.
type Formatter interface {
	// Format formats a document with the given ID, metadata and content.
	Format(docID string, metadata map[string]interface{}, content string) string
}

// AggregateResults aggregates the results of multiple queries into a single result set
// using the given formatter interface to format the results.
func AggregateResults(results []Result, formatter Formatter) string {
	// Sort the results by similarity score in descending order.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Maps docID to accumulated content.
	docContentMap := make(map[string]string)
	// Maps docID to metadata.
	docMetadataMap := make(map[string]map[string]interface{})
	// Maps docID to the highest similarity score.
	docSimilarityMap := make(map[string]float64)

	// Accumulate content and metadata.
	for _, result := range results {
		docID := result.Document.ID
		content, exists := docContentMap[docID]

		if !exists {
			// This is the first time we encounter this document
			docMetadataMap[docID] = result.Document.Metadata
			docContentMap[docID] = result.Segment.Text
			docSimilarityMap[docID] = result.Similarity
		} else {
			// Append the current segment's text to the existing content
			docContentMap[docID] = content + "\n" + result.Segment.Text
		}
	}

	// Create a slice of document IDs sorted by similarity score.
	docIDs := make([]string, 0, len(docSimilarityMap))
	for docID := range docSimilarityMap {
		docIDs = append(docIDs, docID)
	}
	sort.Slice(docIDs, func(i, j int) bool {
		return docSimilarityMap[docIDs[i]] > docSimilarityMap[docIDs[j]]
	})

	// Format the content and metadata for each document and append to the resultBuilder.
	var resultBuilder strings.Builder
	for _, docID := range docIDs {
		metadata := docMetadataMap[docID]
		formattedContent := formatter.Format(docID, metadata, docContentMap[docID])
		resultBuilder.WriteString(formattedContent)
	}

	return resultBuilder.String()
}

// Below are some example implementations of the Formatter interface.

//// StringFormatter is a formatter for formatting the generated report in string format.
//type StringFormatter struct{}
//// Format formats the report based on the given metadata and content in string format.
//func (f *StringFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
//	return fmt.Sprintf("Source: %s\nTitle: %s\nContent: %s\n\n", metadata["url"], metadata["title"], content)
//}
//
//// HTMLFormatter is a formatter for formatting the generated report in HTML format.
//type HTMLFormatter struct{}
//// Format formats the report based on the given metadata and content in HTML format.
//func (f *HTMLFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
//	return fmt.Sprintf(`<p><strong>Source:</strong> %s</p>
//<p><strong>Title:</strong> %s</p>
//<p><strong>Content:</strong> %s</p>
//`, metadata["url"], metadata["title"], content)
//}
//
//// JSONFormatter is a formatter for formatting the generated report in JSON format.
//type JSONFormatter struct{}
//// Format formats the report based on the given metadata and content in JSON format.
//func (f *JSONFormatter) Format(docID string, metadata map[string]interface{}, content string) string {
//	return fmt.Sprintf(`{"source": "%s", "title": "%s", "content": "%s"}`,
//		metadata["url"], metadata["title"], content)
//}
//
//func main() {
//	segments := []vector.Result{
//		// Initialize with your data
//	}
//
//	formatter := &HTMLFormatter{}
//	output := vector.AggregateSegments(segments, formatter)
//	fmt.Println(output)
//}
