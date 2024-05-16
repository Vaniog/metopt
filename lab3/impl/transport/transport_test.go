package transport

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"metopt/ml"
	"testing"
)

func TestModelSerializer_Serialize(t *testing.T) {
	s := NewModelSerializer("./models")
	id, err := s.Serialize(ml.NewLinearModel(ml.Config{
		RowLen: 10,
		Loss:   nil,
		Reg:    nil,
	}))
	assert.NoError(t, err)
	fmt.Println(id)
}

func TestModelSerializer_Deserialize(t *testing.T) {
	id := "41affed5-c5fd-4ece-b8f0-e8f2104714aa"
	s := NewModelSerializer("./models")
	expected := ml.NewLinearModel(ml.Config{RowLen: 10})
	expected.Weights().CopyVec(mat.NewVecDense(10, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
	model, err := s.Deserialize(id)
	assert.NoError(t, err)
	assert.Equal(t, expected, model)
}
