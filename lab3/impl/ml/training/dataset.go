package training

import (
	"gonum.org/v1/gonum/mat"
	"metopt/filter"
	"metopt/ml"
)

type Row struct {
	X mat.Vector
	Y float64
}

type DataSet interface {
	Row(idx int) Row
	Len() int
}

func NewRow(row []float64) Row {
	return Row{
		X: mat.NewVecDense(len(row)-1, row[0:len(row)-1]),
		Y: row[len(row)-1],
	}
}

type SliceDataSet struct {
	Rows []Row
}

func NewSliceDataSet(rows [][]float64) *SliceDataSet {
	return &SliceDataSet{
		Rows: filter.Map(rows, NewRow),
	}
}

func (rd *SliceDataSet) Row(idx int) Row {
	return rd.Rows[idx]
}

func (rd *SliceDataSet) Len() int {
	return len(rd.Rows)
}

func SplitDataSet(ds DataSet) (xs []mat.Vector, ys []float64) {
	for i := range ds.Len() {
		xs = append(xs, ds.Row(i).X)
		ys = append(ys, ds.Row(i).Y)
	}
	return
}

func PredictDataSet(m ml.Model, inputs []mat.Vector) DataSet {
	return &SliceDataSet{
		Rows: filter.Map(inputs, func(row mat.Vector) Row {
			return Row{
				row, m.Predict(row),
			}
		}),
	}
}
