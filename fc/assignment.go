package main

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const nyiTypeFail   = "%s not yet implemented for %T"

func Weightassignment() func () []float32 {
	start := 0
	end := 0
	index := 0
	return func() []float32 {
		end = indexs[index]
		val := weights[start:end+start]
		start = end
		index++
		return val
	}
}

func ValuesOfArray() gorgonia.InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		weight := weight()
		switch dt {
		case tensor.Float64:
			retVal := make([]float64, len(weight))
			for i := range retVal {
				retVal[i] = float64(weight[i])
			}
			return retVal
		case tensor.Float32:
			return weight
		case tensor.Int:
			retVal := make([]int, len(weight))
			for i := range retVal {
				retVal[i] = int(weight[i])
			}
			return retVal
		default:
			err := errors.Errorf(nyiTypeFail, "ValuesOfArray", dt)
			panic(err)
		}
	}
	return f
}


