package training

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestReadCSV(t *testing.T) {
	file, err := ReadCsvFile("../../test-data/test.csv")

	assert.NoError(t, err)
	fmt.Println(file)
}

func TestNewSliceDatasetFromCSV(t *testing.T) {
	expected := NewSliceDataSet([][]float64{
		{10, 10},
		{5, 1},
	})
	ds, err := NewSliceDatasetFromCSV("../../test-data/test.csv")
	assert.NoError(t, err)
	assert.Equal(t, expected, ds)
}
