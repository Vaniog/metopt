package training

import (
	"gonum.org/v1/gonum/mat"
	"metopt/ml"
)

func LossScore(m ml.Model, ds DataSet) float64 {
	curLoss := 0.0
	for i := range ds.Len() {
		r := ds.Row(i)
		curLoss += m.Config().Loss.F(m.Predict(r.X), r.Y)
	}
	return curLoss/float64(ds.Len()) + m.Config().Reg.R(m.Weights())
}

// lossGrad return d(loss)/d(weights) and d(loss)/d(bias)
func lossGrad(m ml.Model, ds DataSet) (*mat.VecDense, float64) {
	gradSum := mat.NewVecDense(m.Weights().Len(), nil)
	biasGradSum := 0.0

	for i := range ds.Len() {
		r := ds.Row(i)
		wGrad := mat.VecDenseCopyOf(m.DP(r.X))
		errorTerm := m.Config().Loss.Df(m.Predict(r.X), r.Y)
		wGrad.ScaleVec(errorTerm, wGrad)
		gradSum.AddVec(gradSum, wGrad)
		if m.Config().Bias {
			biasGradSum += errorTerm
		}

	}

	gradSum.ScaleVec(1.0/float64(ds.Len()), gradSum)
	gradSum.AddVec(gradSum, m.Config().Reg.Dr(m.Weights()))
	biasGradSum /= float64(ds.Len())
	return gradSum, biasGradSum
}
