package main

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const nyiTypeFail = "%s not yet implemented for %T"

func Weightsassignment() func() []float64 {
	start := 0
	end := 0
	item := 0
	return func() []float64 {
		end = WeightsLengthArray[item]
		val := Weightscomposed[start : end+start]
		start += end
		item++
		if item == len(WeightsLengthArray) {
			start = 0
			end = 0
			item = 0
		}
		return val
	}
}

func Weightscomposing(index int, weigt []float64) {
	end := WeightsLengthArray[index]
	for i := 0; i < end; i++ {
		Weightscomposed[WeightsLengthAccu[i]+i] = weigt[i]
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
			retVal := make([]float32, len(weight))
			for i := range retVal {
				retVal[i] = float32(weight[i])
			}
			return retVal
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
