package domain

import "math"

var alpha = 0.01
var c = 0.0

func ChangeAlpha(x float64) {
	alpha = x
}

func ChangeC(x float64) {
	alpha = x
}

func Relu(a float64) float64 {
	if a <= 0 {
		return 0.0
	}
	return a
}

func Relud(a float64) float64 {
	if a >= 0 {
		return 1
	}
	return a
}

func Stup(a float64) float64 {
	if a >= 0.5 {
		return 0.5
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x*(-1)))
}

func Sigmoid_d(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Tanh(x float64) float64 {
	return 2*Sigmoid(2*x) - 1
}

func Tahnd(x float64) float64 {
	return 1 - Tanh(x)*Tanh(x)
}

func Leaky_relu(x float64) float64 {
	if x < 0 {
		return alpha * x
	}
	return x
}

func Leaky_relud(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return alpha
}

func Elu(x float64) float64 {
	if x >= 0 {
		return x
	}
	return alpha * (math.Exp(x) - 1)
}

func Elud(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return Elu(x) * alpha
}

func Linear(x float64) float64 {
	return alpha*x + c
}

func Lineard(x float64) float64 {
	return alpha
}
