package training

import (
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math/rand/v2"
	"metopt/ml"
	"slices"
	"testing"
)

func TestGreedyTrainer_Simple(t *testing.T) {
	ds := NewSliceDataSet([][]float64{
		{0.5, 1},
		{1, 2},
		{2, 4},
		{5, 10},
	})

	trainer := NewGreedyTrainer(
		0.001,
		100000,
		0.001,
	)

	m := ml.NewLinearModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
	})
	trainer.Train(m, ds)
	assert.True(t, LossScore(m, ds) < 0.1)
}

func randFloatSlice(size int, maxAbs float64) []float64 {
	x := make([]float64, size)
	for i := range x {
		x[i] = (rand.Float64() - 0.5) * maxAbs * 2
	}
	return x
}

func genLinearDataSet(coeffs []float64, size int) DataSet {
	data := make([][]float64, 0)

	w := mat.NewVecDense(len(coeffs), coeffs)
	for range size {
		x := randFloatSlice(len(coeffs), 1)
		xV := mat.NewVecDense(len(x), x)
		data = append(data, slices.Concat(x, []float64{mat.Dot(w, xV)}))
	}
	return NewSliceDataSet(data)
}

func FuzzGreedyTrainer_Train(f *testing.F) {
	f.Fuzz(func(t *testing.T, rowLen int) {
		if rowLen > 10 || rowLen <= 0 {
			return
		}
		ds := genLinearDataSet(
			randFloatSlice(rowLen, 1),
			100,
		)

		trainer := NewGreedyTrainer(
			0.001,
			100000,
			0.001,
		)

		m := ml.NewLinearModel(ml.Config{
			RowLen: rowLen,
			Loss:   ml.MSELoss{},
			Reg:    ml.EmptyRegularizator{},
		})
		trainer.Train(m, ds)

		assert.True(t, LossScore(m, ds) < 0.1)
	})
}
