package training

import "metopt/ml"

type GreedyTrainer struct {
	maxIterations int
	targetLoss    float64
	gradStep      float64
}

func NewGreedyTrainer(
	targetLoss float64,
	maxIterations int,
	gradStep float64,
) *GreedyTrainer {
	return &GreedyTrainer{
		targetLoss:    targetLoss,
		maxIterations: maxIterations,
		gradStep:      gradStep,
	}
}

func (gt GreedyTrainer) Train(m ml.Model, ds DataSet) {
	m.Weights().Zero()

	iterations := 0
	for LossScore(m, ds) > gt.targetLoss && iterations < gt.maxIterations {
		grad := lossGrad(m, ds)
		grad.ScaleVec(-1.0*gt.gradStep, grad)
		m.Weights().AddVec(grad, m.Weights())
		iterations++
	}
}
