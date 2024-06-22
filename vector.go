package vector

import (
	"errors"
	"log/slog"
	"math"
	"sort"
	"sync"
)

// normalizeVector normalizes a vector to have unit length.
func normalizeVector(vector []float64) ([]float64, error) {
	var sum float64
	for _, v := range vector {
		sum += v * v
	}

	if sum == 0 {
		return nil, errors.New("cannot normalize a zero vector")
	}

	norm := math.Sqrt(sum)

	normalized := make([]float64, len(vector))
	for i, v := range vector {
		normalized[i] = v / norm
	}

	return normalized, nil
}

const isNormalizedPrecisionTolerance = 1e-6

// isNormalized checks if the vector is normalized.
// We do not use this function in the code because the vectors are already normalized.
func isNormalized(v []float64) bool {
	if len(v) == 0 {
		return false
	}

	var sqSum float64
	for _, val := range v {
		sqSum += val * val
	}
	magnitude := math.Sqrt(sqSum)
	return math.Abs(magnitude-1) < isNormalizedPrecisionTolerance
}

// averageVectors calculates the average of a list of vectors.
func averageVectors(vectors [][]float64) ([]float64, error) {
	if len(vectors) == 0 {
		return nil, errors.New("no vectors to average")
	}
	vecLen := len(vectors[0])
	avgVec := make([]float64, vecLen)
	for _, vec := range vectors {
		if len(vec) != vecLen {
			return nil, errors.New("all vectors must have the same length")
		}
		for i, v := range vec {
			avgVec[i] += v
		}
	}
	for i := range avgVec {
		avgVec[i] /= float64(len(vectors))
	}
	return avgVec, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
// We do not use this function.
// We only support cosine similarity for normalized vectors.
// All documents were already normalized when added to the collection.
func cosineSimilarity(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, errors.New("vectors must have the same length")
	}

	var dotP, normA, normB float64
	for i := range vec1 {
		dotP += vec1[i] * vec2[i]
		normA += vec1[i] * vec1[i]
		normB += vec2[i] * vec2[i]
	}

	if normA == 0 || normB == 0 {
		return 0, errors.New("cannot calculate cosine similarity with zero vectors")
	}

	return dotP / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// dotProduct calculates the dot product between two vectors.
// It's the same as cosine similarity for normalized vectors.
// The result represents the cosine of the angle between the two vectors.
// So, the dot product of two normalized vectors is the cosine similarity.
// A higher dot product means the vectors are more similar.
func dotProduct(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, errors.New("vectors must have the same length")
	}

	var dotP float64
	for i := range vec1 {
		dotP += vec1[i] * vec2[i]
	}

	return dotP, nil
}

// Similarity stores the index and similarity score for sorting.
type Similarity struct {
	ID    string
	Score float64
}

// getTopNSimilarEmbeddings retrieves the top N similar embeddings to the query embedding.
func getTopNSimilarEmbeddings(queryEmbedding []float64, embeddings [][]float64, ids []string, topN int) ([]Similarity, error) {
	if len(embeddings) == 0 {
		return nil, errors.New("no embeddings provided")
	}

	if len(queryEmbedding) != len(embeddings[0]) {
		return nil, errors.New("query embedding length does not match embeddings")
	}

	// Normalize the query embedding.
	normalizedQueryEmbedding, err := normalizeVector(queryEmbedding)
	if err != nil {
		return nil, err
	}

	similarities := make([]Similarity, len(embeddings))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, embedding := range embeddings {
		wg.Add(1)
		go func(i int, embedding []float64) {
			defer wg.Done()
			// Calculate the dot product between the query embedding and each embedding.
			// The dot product is the cosine similarity for normalized vectors.
			score, err := dotProduct(normalizedQueryEmbedding, embedding)
			if err != nil {
				slog.Warn("error calculating dot product for embedding", "index", i, "error", err)
				return
			}

			mu.Lock()
			similarities[i] = Similarity{
				ID:    ids[i],
				Score: score,
			}
			mu.Unlock()
		}(i, embedding)
	}

	wg.Wait()

	// Sort the similarities by score in descending order.
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Score > similarities[j].Score
	})

	if topN > len(similarities) {
		topN = len(similarities)
	}

	return similarities[:topN], nil
}
