package main

import (
	"fmt"
	"log"
	"math"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type RestBlockOpts struct {
	InputDimension  int
	OutputDimension int
	Activation      activation.Function
	KernelSize1x1   tensor.Shape
	KernelSize3x3   tensor.Shape
	Pad0            []int
	Pad1            []int
	Stride          []int
	Dilation        []int
	WeightsInit1x1  gorgonia.InitWFn
	WeightsInit3x3  gorgonia.InitWFn
	FixedWeights    bool
	Layer           int
	Stage           int
	Block           int
}

func (o *RestBlockOpts) setDefaults() {
	o.FixedWeights = false

	if o.Activation == nil {
		o.Activation = activation.Rectify
	}

	if o.KernelSize1x1 == nil {
		o.KernelSize1x1 = tensor.Shape{1, 1}
	}
	if o.KernelSize3x3 == nil {
		o.KernelSize3x3 = tensor.Shape{3, 3}
	}

	if o.Pad0 == nil {
		o.Pad0 = []int{0, 0}
	}

	if o.Pad1 == nil {
		o.Pad1 = []int{1, 1}
	}

	if o.Stride == nil {
		o.Stride = []int{1, 1}
	}

	if o.Dilation == nil {
		o.Dilation = []int{1, 1}
	}

	if o.WeightsInit1x1 == nil {
		if WeightsLengthSum == 0 {
			k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize1x1[0]*o.KernelSize1x1[1]))
			o.WeightsInit1x1 = gorgonia.Uniform(-k, k)
		} else {
			o.WeightsInit1x1 = ValuesOfArray()
		}
	}

	if o.WeightsInit3x3 == nil {
		if WeightsLengthSum == 0 {
			k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize3x3[0]*o.KernelSize3x3[1]))
			o.WeightsInit3x3 = gorgonia.Uniform(-k, k)
		} else {
			o.WeightsInit1x1 = ValuesOfArray()
		}
	}

}

type RestBlockModule struct {
	model  *Model
	layer  godl.LayerType
	opts   RestBlockOpts
	bns    []*BatchNormModule
	weight []*godl.Node
}

func (m *RestBlockModule) Name() string {
	return "RestNetBlock"
}

func (m *RestBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	XX := gorgonia.Must(gorgonia.Conv2d(inputs[0], m.weight[3], m.opts.KernelSize1x1, m.opts.Pad0, m.opts.Stride, m.opts.Dilation))
	Result := m.bns[3].Forward(XX)

	x := gorgonia.Must(gorgonia.Conv2d(inputs[0], m.weight[0], m.opts.KernelSize1x1, m.opts.Pad0, []int{1, 1}, m.opts.Dilation))
	result := m.bns[0].Forward(x)

	x = gorgonia.Must(gorgonia.Conv2d(result[0], m.weight[1], m.opts.KernelSize3x3, m.opts.Pad1, m.opts.Stride, m.opts.Dilation))
	result = m.bns[1].Forward(x)

	x = gorgonia.Must(gorgonia.Conv2d(result[0], m.weight[2], m.opts.KernelSize1x1, m.opts.Pad0, []int{1, 1}, m.opts.Dilation))
	result = m.bns[2].Forward(x)

	x, err := gorgonia.Add(result[0], Result[0])
	if err != nil {
		log.Printf("Adding the identity is failed with Shape of %s!", x.Shape())
	}

	x = gorgonia.Must(m.opts.Activation(x))
	fmt.Println("The shape end of this block is:", x.Shape())
	return godl.Nodes{x}
}

func RestBlock(m *Model, opts RestBlockOpts) *RestBlockModule {
	opts.setDefaults()
	lt := godl.AddLayer("RestNet.Restblock")

	ConvBlockName := fmt.Sprintf("Convulution:%d_%d", opts.Layer, opts.Stage)
	w1 := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, 1, 1}, NewWeightsOpts{
		InitFN:     opts.WeightsInit1x1,
		UniqueName: ConvBlockName + "_1",
		Fixed:      opts.FixedWeights,
	})
	w2 := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.OutputDimension, 3, 3}, NewWeightsOpts{
		InitFN:     opts.WeightsInit3x3,
		UniqueName: ConvBlockName + "_2",
		Fixed:      opts.FixedWeights,
	})
	w3 := m.AddWeights(lt, tensor.Shape{opts.OutputDimension * 4, opts.OutputDimension, 1, 1}, NewWeightsOpts{
		InitFN:     opts.WeightsInit1x1,
		UniqueName: ConvBlockName + "_3",
		Fixed:      opts.FixedWeights,
	})
	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension * 4, opts.InputDimension, 1, 1}, NewWeightsOpts{
		InitFN:     opts.WeightsInit1x1,
		UniqueName: ConvBlockName + "_1_rest",
		Fixed:      opts.FixedWeights,
	})

	BatchNormName := fmt.Sprintf("Normalize:%d_%d", opts.Layer, opts.Stage)
	bn1 := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension,
		ScaleName: BatchNormName + "_1/gamma",
		BiasName:  BatchNormName + "_1/beta",
	})
	bn2 := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension,
		ScaleName: BatchNormName + "_2/gamma",
		BiasName:  BatchNormName + "_2/beta",
	})
	bn3 := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension * 4,
		ScaleName: BatchNormName + "_3/gamma",
		BiasName:  BatchNormName + "_3/beta",
	})
	bn := BatchNorm2d(m, BatchNormOpts{
		InputSize: opts.OutputDimension * 4,
		ScaleName: BatchNormName + "_1/gamma_rest",
		BiasName:  BatchNormName + "_1/beta_rest",
	})

	return &RestBlockModule{
		model:  m,
		layer:  lt,
		opts:   opts,
		weight: []*godl.Node{w1, w2, w3, w},
		bns:    []*BatchNormModule{bn1, bn2, bn3, bn},
	}
}

var (
	_ godl.Module = &RestBlockModule{}
)
