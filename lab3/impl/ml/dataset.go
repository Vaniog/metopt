package ml

import "gonum.org/v1/gonum/mat"

type Row struct {
	X mat.Vector
	Y float64
}

type DataSet interface {
	Row(idx int) Row
	Len() int
}

type RowDataSet struct {
	Rows []Row
}

func NewRowDataSet(rows [][]float64) *RowDataSet {
	rd := &RowDataSet{make([]Row, len(rows))}
	for i := range rows {
		row := rows[i]
		rd.Rows[i] = Row{
			X: mat.NewVecDense(len(row)-1, row[0:len(row)-1]),
			Y: row[len(row)-1],
		}
	}
	return rd
}

func (rd *RowDataSet) Row(idx int) Row {
	return rd.Rows[idx]
}

func (rd *RowDataSet) Len() int {
	return len(rd.Rows)
}
