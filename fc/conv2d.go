package main

import (
	"fmt"
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Conv2dOpts are the options to run the conv2d operation
type Conv2dOpts struct {
	InputDimension  int
	OutputDimension int

	KernelSize tensor.Shape
	Pad        []int
	Stride     []int
	Dilation   []int

	WithBias bool

	WeightsInit, BiasInit gorgonia.InitWFn
	WeightsName, BiasName string
	FixedWeights          bool
}

func (o *Conv2dOpts) setDefaults() {
	if o.KernelSize == nil {
		o.KernelSize = tensor.Shape{3, 3}
	}

	if o.Pad == nil {
		o.Pad = []int{0, 0}
	}

	if o.Stride == nil {
		o.Stride = []int{1, 1}
	}

	if o.Dilation == nil {
		o.Dilation = []int{1, 1}
	}

	if o.WeightsInit == nil {
		k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}

	if o.BiasInit == nil {
		k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}
}

type Conv2dModule struct {
	model *godl.Model
	layer godl.LayerType

	opts Conv2dOpts

	weight, bias *godl.Node
}

func (m *Conv2dModule) Name() string {
	return "Conv2d"
}

func (m *Conv2dModule) Forward(inputs ...*godl.Node) godl.Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}
	x := inputs[0]
	fmt.Printf("Conv2d: weights is %v, shape is %v, 1 is %v, 2 is %v, 3 is %v \n", m.weight.Shape(), m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation)
	x = gorgonia.Must(gorgonia.Conv2d(x, m.weight, m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation))

	if m.bias != nil {
		x = gorgonia.Must(gorgonia.BroadcastAdd(x, m.bias, nil, []byte{0, 2, 3}))
	}

	return godl.Nodes{x}
}

// Conv2d applies a conv2d operation to the input
func Conv2d(m *godl.Model, opts Conv2dOpts) *Conv2dModule {
	opts.setDefaults()
	lt := godl.AddLayer("Conv2d")

	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, godl.NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: opts.WeightsName,
		Fixed:      opts.FixedWeights,
	})

	var bias *gorgonia.Node
	if opts.WithBias {
		bias = m.AddBias(lt, tensor.Shape{1, opts.OutputDimension, 1, 1}, godl.NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	return &Conv2dModule{
		model:  m,
		layer:  lt,
		opts:   opts,
		weight: w,
		bias:   bias,
	}
}

var (
	_ godl.Module = &Conv2dModule{}
)
