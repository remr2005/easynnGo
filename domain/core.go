package domain

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type ActivationFunc func(float64) float64

type NeuralNetwork struct {
	alpha          float64
	layers         []mat.Dense
	weights        []mat.Dense
	activ_func     ActivationFunc
	dev_activ_func ActivationFunc
	exitfunction   ActivationFunc
}

// Create new neural network, accept activate function
func NewNeuralNetwork(alpha float64, a ActivationFunc, ad ActivationFunc, exitfunction ActivationFunc) NeuralNetwork {
	var res NeuralNetwork
	res.alpha = alpha
	res.activ_func = a
	res.dev_activ_func = ad
	res.exitfunction = exitfunction
	res.layers = make([]mat.Dense, 0)
	res.weights = make([]mat.Dense, 0)
	return res
}

// Add input layer
func (A *NeuralNetwork) AddInputLayer(a int) error {
	if len(A.layers) != 0 {
		return fmt.Errorf("input layer already exist")
	}
	var Buf_Matrix = mat.NewDense(1, a, nil)
	A.layers = append(A.layers, *Buf_Matrix)
	return nil
}

// Add hiden layer
func (A *NeuralNetwork) AddHideLayer(a int) error {
	if len(A.layers) == 0 {
		return fmt.Errorf("input layer doesn't exist")
	}
	layer_Matrix := mat.NewDense(1, a, nil)
	_, c := A.layers[len(A.layers)-1].Dims()
	buf := make([]float64, 0)
	m := rand.New(rand.NewSource(time.Now().Unix()))
	for i := 0; i < a*c; i++ {
		buf = append(buf, m.Float64())
	}
	weight_Matrix := mat.NewDense(a, c, buf)
	A.layers = append(A.layers, *layer_Matrix)
	A.weights = append(A.weights, *weight_Matrix)
	return nil
}

func (A *NeuralNetwork) AddDropOut() {

}

func (A *NeuralNetwork) Train(epoch int, x_train []mat.Dense, y_train []mat.Dense) {
	for i := 0; i < epoch; i++ {
		a := rand.New(rand.NewSource(time.Now().Unix()))
		b := a.Int() % len(x_train)
		A.layers[0] = x_train[b]
		for j := 0; j < len(A.weights); j++ {
			A.layers[j+1].Product(&A.layers[j], A.weights[j].T())
		}
		err := y_train[b]
		for j := len(A.weights); j > 0; j-- {
			err.Sub(&A.layers[j], &err)
			_, c_err := err.Dims()
			_, c_inp := A.layers[j-1].Dims()
			delta := mat.NewDense(c_err, c_inp, nil)
			delta.Product(err.T(), &A.layers[j-1])
			delta.Scale(A.alpha, delta)
			A.weights[j-1].Sub(&A.weights[j-1], delta)
			err.Product(&err, &A.weights[j-1])
		}
	}
}

func (A *NeuralNetwork) Return(x mat.Dense) mat.Dense {
	A.layers[0] = x
	for j := 0; j < len(A.weights); j++ {
		A.layers[j+1].Product(&A.layers[j], A.weights[j].T())
	}
	return A.layers[len(A.layers)-1]
}
