package training

import (
	"encoding/csv"
	"errors"
	"gonum.org/v1/gonum/mat"
	"metopt/filter"
	"os"
	"strconv"
)

var ErrEmptyFile = errors.New("empty file")
var ErrBadData = errors.New("bad data")

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

func ReadCsvFile(filePath string) ([][]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}

func parseRow(row []string) (Row, error) {
	train, err := filter.MapWithError(row,
		func(s string) (float64, error) {
			return strconv.ParseFloat(s, 32)
		})
	if err != nil {
		return Row{}, err
	}
	return NewRow(train), nil
}

func NewSliceDatasetFromCSV(path string) (DataSet, error) {
	raw, err := ReadCsvFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, ErrEmptyFile
	}
	if len(raw[0]) < 2 {
		return nil, ErrBadData
	}

	rows := make([]Row, len(raw))
	for i, row := range raw {
		rows[i], err = parseRow(row)

	}
	return &SliceDataSet{
		Rows: rows,
	}, err
}
