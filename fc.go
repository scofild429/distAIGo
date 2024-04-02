package main

import (
	"fmt"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type LinearOpts struct {
	Activation      activation.Function
	Dropout         float64
	OutputDimension int
	InputDimension  int

	WeightsInit           gorgonia.InitWFn
	BiasInit              gorgonia.InitWFn
	WithBias              bool
	WeightsName, BiasName string
	FixedWeights          bool
}

type LinearModule struct {
	model *Model
	opts  LinearOpts
	layer godl.LayerType

	weight, bias *godl.Node
}

func (m *LinearModule) Name() string {
	return "Linear"
}

func (m *LinearModule) Forward(inputs ...*godl.Node) (out godl.Nodes) {
	x := inputs[0]

	x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{7, 7}, []int{0, 0}, []int{1, 1}))
	fmt.Println("the shape after the global max pooling :", x.Shape())
	xShape := x.Shape()

	if x.Dims() > 2 {
		b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
		x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
	}

	wT := gorgonia.Must(gorgonia.Transpose(m.weight, 1, 0))

	result, err := gorgonia.Mul(x, wT)
	if err != nil {
		panic(godl.ErrorF(m.layer, "error applying mul %v x %v: %w ", x.Shape(), wT.Shape(), err))
	}

	if m.opts.WithBias {
		result, err = gorgonia.BroadcastAdd(result, m.bias, nil, []byte{0})
		if err != nil {
			panic(godl.ErrorF(m.layer, "error adding bias %w", err))
		}
	}

	if m.opts.Activation != nil {
		result, err = m.opts.Activation(result)
		if err != nil {
			panic(godl.ErrorF(m.layer, "error applying activation %w", err))
		}
	}

	if m.opts.Dropout > 0.0 {
		result, err = gorgonia.Dropout(result, m.opts.Dropout)
		if err != nil {
			panic(godl.ErrorF(m.layer, "error applying dropout %w", err))
		}
	}

	return godl.Nodes{result}
}

func Linear(nn *Model, opts LinearOpts) *LinearModule {
	lt := godl.AddLayer("FC")

	godl.MustBeGreatherThan(lt, "input dimension", opts.InputDimension, 0)
	godl.MustBeGreatherThan(lt, "output dimension", opts.OutputDimension, 0)

	var (
		bias *gorgonia.Node
		w    = nn.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension}, NewWeightsOpts{
			InitFN:     opts.WeightsInit,
			UniqueName: opts.WeightsName,
			Fixed:      opts.FixedWeights,
		})
	)

	if opts.WithBias {
		bias = nn.AddBias(lt, tensor.Shape{1, opts.OutputDimension}, NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	//	nn.Watch("The last node", w)

	return &LinearModule{
		model:  nn,
		layer:  lt,
		opts:   opts,
		bias:   bias,
		weight: w,
	}
}

var (
	_ godl.Module = &LinearModule{}
)
