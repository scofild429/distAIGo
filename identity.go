package main

import (
	"fmt"
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type IdentityOpts struct {
	InputDimension  int
	OutputDimension int
	KernelSize      tensor.Shape
	Pad             []int
	Stride          []int
	Dilation        []int
	WeightsInit     gorgonia.InitWFn
	FixedWeights    bool
	Layer           int
}

func (o *IdentityOpts) setDefaults() {
	o.FixedWeights = false

	if o.KernelSize == nil {
		o.KernelSize = tensor.Shape{1, 1}
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
}

type IdentityModule struct {
	model     *Model
	layer     godl.LayerType
	opts      IdentityOpts
	bns       *BatchNormModule
	weight    *godl.Node
	expansion int
}

func (m *IdentityModule) Name() string {
	return "IdentityOfBlock"
}

func (m *IdentityModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := gorgonia.Must(gorgonia.Conv2d(inputs[0], m.weight, m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation))
	result := m.bns.Forward(x)

	return godl.Nodes{result[0]}
}

func Identity(m *Model, opts IdentityOpts) *IdentityModule {
	opts.setDefaults()
	lt := godl.AddLayer("RestNet.Identity")

	ConvBlockName := fmt.Sprintf("Convulution:%d_", opts.Layer)
	w1 := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: ConvBlockName + "_1",
		Fixed:      opts.FixedWeights,
	})

	BatchNormName := fmt.Sprintf("Normalize:%d_", opts.Layer)
	bn1 := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension,
		ScaleName: BatchNormName + "_1/gamma",
		BiasName:  BatchNormName + "_1/beta",
	})
	return &IdentityModule{
		model:  m,
		layer:  lt,
		opts:   opts,
		weight: w1,
		bns:    bn1,
	}
}

var (
	_ godl.Module = &IdentityModule{}
)
