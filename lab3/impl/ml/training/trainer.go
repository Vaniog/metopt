package training

import "metopt/ml"

type Trainer interface {
	Train(ml.Model, DataSet)
}
