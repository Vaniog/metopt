package ml

import (
	"gonum.org/v1/gonum/mat"
)

type Trainer interface {
	Train(Model, DataSet)
}

type GreedyTrainer struct {
	loss          Loss
	reg           Regularizator
	maxIterations int
	targetLoss    float64
	gradStep      float64
}

func NewGreedyTrainer(
	l Loss,
	r Regularizator,
	targetLoss float64,
	maxIterations int,
	gradStep float64,
) *GreedyTrainer {
	return &GreedyTrainer{
		loss:          l,
		reg:           r,
		targetLoss:    targetLoss,
		maxIterations: maxIterations,
		gradStep:      gradStep,
	}
}

func (gt GreedyTrainer) Train(m Model, ds DataSet) {
	m.weights().Zero()

	iterations := 0
	for LossScore(m, ds, gt.loss) > gt.targetLoss && iterations < gt.maxIterations {
		grad := lossGrad(m, ds, gt.loss, gt.reg)
		grad.ScaleVec(-1.0*gt.gradStep, grad)
		m.weights().AddVec(grad, m.weights())
		iterations++
	}
}

func LossScore(m Model, ds DataSet, loss Loss) float64 {
	curLoss := 0.0
	for i := range ds.Len() {
		r := ds.Row(i)
		curLoss += loss.F(m.Predict(r.X), r.Y)
	}
	return curLoss / float64(ds.Len())
}

func lossGrad(m Model, ds DataSet, loss Loss, reg Regularizator) *mat.VecDense {
	gradSum := mat.NewVecDense(ds.Row(0).X.Len(), nil)

	for i := range ds.Len() {
		r := ds.Row(i)
		wGrad := mat.VecDenseCopyOf(m.dw(r.X))
		wGrad.ScaleVec(loss.Df(m.Predict(r.X), r.Y), wGrad)
		gradSum.AddVec(gradSum, wGrad)
	}

	gradSum.ScaleVec(1.0/float64(ds.Len()), gradSum)
	gradSum.AddVec(gradSum, reg.Dr(m.weights()))
	return gradSum
}
