package vector

import (
	"math"
	"testing"
)

func TestNormalizeVector(t *testing.T) {
	testCases := []struct {
		name     string
		input    []float64
		expected []float64
		wantErr  bool
	}{
		{
			name:     "Normal vector",
			input:    []float64{3, 4},
			expected: []float64{0.6, 0.8},
			wantErr:  false,
		},
		{
			name:     "Zero vector",
			input:    []float64{0, 0},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Negative values",
			input:    []float64{-1, -1},
			expected: []float64{-0.7071067811865475, -0.7071067811865475},
			wantErr:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			normalized, err := normalizeVector(tc.input)
			if (err != nil) != tc.wantErr {
				t.Fatalf("expected error: %v, got: %v", tc.wantErr, err)
			}
			if !tc.wantErr {
				for i, v := range normalized {
					if math.Abs(v-tc.expected[i]) > 1e-9 {
						t.Errorf("expected %v, got %v", tc.expected, normalized)
						break
					}
				}
			}
		})
	}
}

func TestIsNormalized(t *testing.T) {
	testCases := []struct {
		name     string
		input    []float64
		expected bool
	}{
		{
			name:     "Normalized vector",
			input:    []float64{0.6, 0.8},
			expected: true,
		},
		{
			name:     "Unnormalized vector",
			input:    []float64{1, 1},
			expected: false,
		},
		{
			name:     "Zero vector",
			input:    []float64{0, 0},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := isNormalized(tc.input)
			if result != tc.expected {
				t.Errorf("expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestAverageVectors(t *testing.T) {
	testCases := []struct {
		name     string
		input    [][]float64
		expected []float64
		wantErr  bool
	}{
		{
			name:     "Equal length vectors",
			input:    [][]float64{{1, 2}, {3, 4}},
			expected: []float64{2, 3},
			wantErr:  false,
		},
		{
			name:     "Different length vectors",
			input:    [][]float64{{1, 2}, {3, 4, 5}},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Empty input",
			input:    [][]float64{},
			expected: nil,
			wantErr:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			avg, err := averageVectors(tc.input)
			if (err != nil) != tc.wantErr {
				t.Fatalf("expected error: %v, got: %v", tc.wantErr, err)
			}
			if !tc.wantErr {
				for i, v := range avg {
					if math.Abs(v-tc.expected[i]) > 1e-9 {
						t.Errorf("expected %v, got %v", tc.expected, avg)
						break
					}
				}
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	testCases := []struct {
		name     string
		vec1     []float64
		vec2     []float64
		expected float64
		wantErr  bool
	}{
		{
			name:     "Equal length vectors",
			vec1:     []float64{1, 2, 3},
			vec2:     []float64{4, 5, 6},
			expected: 32,
			wantErr:  false,
		},
		{
			name:     "Different length vectors",
			vec1:     []float64{1, 2},
			vec2:     []float64{4, 5, 6},
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "Zero vectors",
			vec1:     []float64{0, 0},
			vec2:     []float64{0, 0},
			expected: 0,
			wantErr:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dot, err := dotProduct(tc.vec1, tc.vec2)
			if (err != nil) != tc.wantErr {
				t.Fatalf("expected error: %v, got: %v", tc.wantErr, err)
			}
			if !tc.wantErr && math.Abs(dot-tc.expected) > 1e-9 {
				t.Errorf("expected %v, got %v", tc.expected, dot)
			}
		})
	}
}

func TestGetTopNSimilarEmbeddings(t *testing.T) {
	testCases := []struct {
		name           string
		queryEmbedding []float64
		embeddings     [][]float64
		ids            []string
		topN           int
		expected       []Similarity
		wantErr        bool
	}{
		{
			name:           "Normal case",
			queryEmbedding: []float64{0.6, 0.8},
			embeddings:     [][]float64{{0.6, 0.8}, {0.8, 0.6}},
			ids:            []string{"doc1", "doc2"},
			topN:           2,
			expected: []Similarity{
				{ID: "doc1", Score: 1.0},
				{ID: "doc2", Score: 0.96},
			},
			wantErr: false,
		},
		{
			name:           "Different length vectors",
			queryEmbedding: []float64{0.6, 0.8},
			embeddings:     [][]float64{{0.6, 0.8, 0.0}},
			ids:            []string{"doc1"},
			topN:           1,
			expected:       nil,
			wantErr:        true,
		},
		{
			name:           "Empty embeddings",
			queryEmbedding: []float64{0.6, 0.8},
			embeddings:     [][]float64{},
			ids:            []string{},
			topN:           1,
			expected:       nil,
			wantErr:        true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			topN, err := getTopNSimilarEmbeddings(tc.queryEmbedding, tc.embeddings, tc.ids, tc.topN)
			if (err != nil) != tc.wantErr {
				t.Fatalf("expected error: %v, got: %v", tc.wantErr, err)
			}
			if !tc.wantErr {
				for i, sim := range topN {
					if sim.ID != tc.expected[i].ID || math.Abs(sim.Score-tc.expected[i].Score) > 1e-9 {
						t.Errorf("expected %v, got %v", tc.expected, topN)
						break
					}
				}
			}
		})
	}
}
