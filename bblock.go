package main

import (
	"fmt"
	"math"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type BBlockOpts struct {
	InputDimension        int
	OutputDimension       int
	Activation            activation.Function
	Dropout               float64
	KernelSize            tensor.Shape
	Pad                   []int
	Stride                []int
	Dilation              []int
	WithBias              bool
	WithPooling           bool
	WeightsInit, BiasInit gorgonia.InitWFn
	WeightsName, BiasName string
	FixedWeights          bool
	Stage                 int
	Block                 int
	Layer                 int
}

func (o *BBlockOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = activation.Rectify
	}

	if o.KernelSize == nil {
		o.KernelSize = tensor.Shape{3, 3}
	}

	if o.Pad == nil {
		o.Pad = []int{1, 1}
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

type BBlockModule struct {
	model        *Model
	layer        godl.LayerType
	opts         BBlockOpts
	bns          []*BatchNormModule
	weight, bias *godl.Node
}

func (m *BBlockModule) Name() string {
	return "RestNetBlock"
}

func (m *BBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	x = gorgonia.Must(gorgonia.Conv2d(x, m.weight, m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation))
	result := m.bns[0].Forward(x)
	x = result[0]

	if m.bias != nil {
		x = gorgonia.Must(gorgonia.BroadcastAdd(x, m.bias, nil, []byte{0, 2, 3}))
	}

	if m.opts.Activation != nil {
		x = gorgonia.Must(m.opts.Activation(x))
	}

	if m.opts.WithPooling {
		x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}))
	}

	if m.opts.Dropout > 0.0 {
		x = gorgonia.Must(gorgonia.Dropout(x, m.opts.Dropout))
	}

	fmt.Println("the shape after the block :", x.Shape())

	return godl.Nodes{x}
}

// Block is a VGG block composed of conv2d+maxpool with optional dropout and activation function
func BBlock(m *Model, opts BBlockOpts) *BBlockModule {
	opts.setDefaults()

	lt := godl.AddLayer("RestNet.Block")
	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: opts.WeightsName,
		Fixed:      opts.FixedWeights,
	})

	var bias *gorgonia.Node
	if opts.WithBias {
		bias = m.AddBias(lt, tensor.Shape{1, opts.OutputDimension, 1, 1}, NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	BatchNormName := fmt.Sprintf("Normalize%d_%d", opts.Stage, opts.Block)
	bn1 := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension,
		ScaleName: BatchNormName + "/bn/gamma",
		BiasName:  BatchNormName + "/bn/beta",
	})

	return &BBlockModule{
		model:  m,
		layer:  lt,
		opts:   opts,
		weight: w,
		bias:   bias,
		bns:    []*BatchNormModule{bn1},
	}
}

var (
	_ godl.Module = &BBlockModule{}
)
