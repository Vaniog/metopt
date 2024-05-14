package ml

import (
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestGreedyTrainer_Simple(t *testing.T) {
	ds := NewRowDataSet([][]float64{
		{0.5, 1},
		{1, 2},
		{2, 4},
		{5, 10},
	})

	trainer := NewGreedyTrainer(
		MSELoss{},
		EmptyRegularizator{},
		0.001,
		100000,
		0.001,
	)

	m := NewLinearModel(1)
	trainer.Train(m, ds)
	assert.True(t, LossScore(m, ds, MSELoss{}) < 0.1)
}

func randFloatSlice(size int, maxAbs float64) []float64 {
	x := make([]float64, size)
	for i := range x {
		x[i] = (rand.Float64() - 0.5) * maxAbs * 2
	}
	return x
}

func genLinearDataSet(coeffs []float64, size int) [][]float64 {
	data := make([][]float64, 0)

	w := mat.NewVecDense(len(coeffs), coeffs)
	for range size {
		x := randFloatSlice(len(coeffs), 1)
		xV := mat.NewVecDense(len(x), x)
		data = append(data, slices.Concat(x, []float64{mat.Dot(w, xV)}))
	}
	return data
}

func FuzzGreedyTrainer_Train(f *testing.F) {
	f.Fuzz(func(t *testing.T, rowLen int) {
		if rowLen > 10 || rowLen <= 0 {
			return
		}
		ds := NewRowDataSet(genLinearDataSet(
			randFloatSlice(rowLen, 1),
			100,
		))

		trainer := NewGreedyTrainer(
			MSELoss{},
			EmptyRegularizator{},
			0.001,
			100000,
			0.001,
		)

		m := NewLinearModel(rowLen)
		trainer.Train(m, ds)

		assert.True(t, LossScore(m, ds, MSELoss{}) < 0.1)
	})
}
