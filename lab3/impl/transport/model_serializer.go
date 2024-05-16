package transport

import (
	"encoding/json"
	"github.com/google/uuid"
	"gonum.org/v1/gonum/mat"
	"metopt/ml"
	"metopt/transport/generated"
	"os"
	"path"
)

type ModelSerializer struct {
	path string
}

func NewModelSerializer(path string) *ModelSerializer {
	return &ModelSerializer{
		path: path,
	}
}

func (s *ModelSerializer) Serialize(model ml.Model) (string, error) {
	transportModel := &generated.Model{Weights: model.Weights().RawVector().Data}
	id := uuid.New()
	serialized, err := json.Marshal(transportModel)
	if err != nil {
		return "", err
	}
	err = s.writeToFile(id.String(), serialized)
	if err != nil {
		return id.String(), err
	}
	return id.String(), nil
}

func (s *ModelSerializer) Deserialize(id string) (ml.Model, error) {
	raw, err := s.readFromFile(id)
	if err != nil {
		return nil, err
	}
	var transportModel generated.Model

	err = json.Unmarshal(raw, &transportModel)
	if err != nil {
		return nil, err
	}
	model := ml.NewLinearModel(ml.Config{RowLen: len(transportModel.Weights)})
	model.Weights().CopyVec(mat.NewVecDense(len(transportModel.Weights), transportModel.Weights))
	return model, nil
}

func (s *ModelSerializer) readFromFile(id string) ([]byte, error) {
	return os.ReadFile(path.Join(s.path, id))
}
func (s *ModelSerializer) writeToFile(id string, data []byte) error {
	return os.WriteFile(path.Join(s.path, id), data, 0777)
}
