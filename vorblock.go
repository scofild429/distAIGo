package main

import (
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type VorBlockOpts struct {
	InputDimension         int
	OutputDimension        int
	KernelSize             tensor.Shape
	Pad                    []int
	Stride                 []int
	Dilation               []int
	WithBias               bool
	WeightsInit, wBiasInit gorgonia.InitWFn
	WeightsName, wBiasName string
	FixedWeights           bool
	InputSize              int
	Momentum               float64
	Epsilon                float64
	ScaleInit, bBiasInit   gorgonia.InitWFn
	ScaleName, bBiasName   string
}

func (o *VorBlockOpts) setDefaults() {
	if o.InputSize == 0 {
		o.InputSize = o.OutputDimension
	}

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
		if WeightsLengthSum == 0 {
			k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
			o.WeightsInit = gorgonia.Uniform(-k, k)
		} else {
			o.WeightsInit = ValuesOfArray()
		}
	}

	if o.wBiasInit == nil {
		if WeightsLengthSum == 0 {
			k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
			o.WeightsInit = gorgonia.Uniform(-k, k)
		} else {
			o.WeightsInit = ValuesOfArray()
		}
	}

	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.ScaleInit == nil {
		if WeightsLengthSum == 0 {
			o.ScaleInit = gorgonia.Ones()
		} else {
			o.ScaleInit = ValuesOfArray()
		}
	}

	if o.bBiasInit == nil {
		if WeightsLengthSum == 0 {
			o.bBiasInit = gorgonia.Zeroes()
		} else {
			o.bBiasInit = ValuesOfArray()
		}
	}
}

type VorBlockModule struct {
	model       *Model
	layer       godl.LayerType
	opts        VorBlockOpts
	weight      *godl.Node
	scale, bias *godl.Node
}

func (m *VorBlockModule) Name() string {
	return "RestNetVorBlock"
}

func (m *VorBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}
	x := inputs[0]

	fmt.Printf("In Vorblock Conv2d: weights is %v, shape is %v, 1 is %v, 2 is %v, 3 is %v \n", m.weight.Shape(), m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation)
	x = gorgonia.Must(gorgonia.Conv2d(x, m.weight, m.opts.KernelSize, m.opts.Pad, m.opts.Stride, m.opts.Dilation))
	ret, _, _, _, err := gorgonia.BatchNorm(x, m.scale, m.bias, float64(m.opts.Momentum), float64(m.opts.Epsilon))
	x = gorgonia.Must(gorgonia.Rectify(ret))
	x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{3, 3}, []int{1, 1}, []int{2, 2}))

	return godl.Nodes{x}
}

func VorBlock(m *Model, opts VorBlockOpts) *VorBlockModule {
	opts.setDefaults()

	lt := godl.AddLayer("RestNet.VorBlock")
	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: opts.WeightsName,
		Fixed:      opts.FixedWeights,
	})

	batchSize, _ := strconv.Atoi(os.Getenv("batchSize"))
	scale := m.AddLearnable(lt, "scale", tensor.Shape{batchSize, opts.InputSize, 1, 1}, NewWeightsOpts{
		UniqueName: opts.ScaleName,
		InitFN:     opts.ScaleInit,
	})
	bias := m.AddBias(lt, tensor.Shape{batchSize, opts.InputSize, 1, 1}, NewWeightsOpts{
		UniqueName: opts.bBiasName,
		InitFN:     opts.bBiasInit,
	})

	return &VorBlockModule{
		model:  m,
		layer:  lt,
		opts:   opts,
		weight: w,
		scale:  scale,
		bias:   bias,
	}
}

var (
	_ godl.Module = &VorBlockModule{}
)
