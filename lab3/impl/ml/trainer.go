package ml

type Trainer interface {
	Train(Dataset) Model
}
