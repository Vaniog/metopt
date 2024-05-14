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

func lossGrad(m ml.Model, ds DataSet) *mat.VecDense {
	gradSum := mat.NewVecDense(ds.Row(0).X.Len(), nil)

	for i := range ds.Len() {
		r := ds.Row(i)
		wGrad := mat.VecDenseCopyOf(m.DP(r.X))
		wGrad.ScaleVec(m.Config().Loss.Df(m.Predict(r.X), r.Y), wGrad)
		gradSum.AddVec(gradSum, wGrad)
	}

	gradSum.ScaleVec(1.0/float64(ds.Len()), gradSum)
	gradSum.AddVec(gradSum, m.Config().Reg.Dr(m.Weights()))
	return gradSum
}
