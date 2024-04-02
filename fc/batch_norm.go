package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// BatchNormOpts are the options to configure a batch normalization
type BatchNormOpts struct {
	Momentum            float64
	Epsilon             float64
	ScaleInit, BiasInit gorgonia.InitWFn
	ScaleName, BiasName string
	InputSize int
}

func (o *BatchNormOpts) setDefaults() {
	if o.InputSize == 0 {
		panic("output size for BN can't be 0")
	}

	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.ScaleInit == nil {
		o.ScaleInit = ValuesOfArray()
	}

	if o.BiasInit == nil {
		o.BiasInit =  ValuesOfArray()
	}
}

type BatchNormModule struct {
	model *Model
	layer godl.LayerType
	opts  BatchNormOpts
	scale, bias *godl.Node
}

func (m *BatchNormModule) Name() string {
	return "BatchNorm"
}

func (m *BatchNormModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	ret, _, _, _, err := gorgonia.BatchNorm(x, m.scale, m.bias, float64(m.opts.Momentum), float64(m.opts.Epsilon))
	if err != nil {
		panic(fmt.Errorf("%v: %w", m.layer, err))
	}

	return godl.Nodes{ret}
}

// BatchNorm1d defines the batch norm operation for tensors with shape (B, N)
func BatchNorm1d(nn *Model, opts BatchNormOpts) *BatchNormModule {
	opts.setDefaults()
	lt := godl.AddLayer("BatchNorm1d")

	scale := nn.AddLearnable(lt, "scale", tensor.Shape{1, opts.InputSize}, NewWeightsOpts{
		UniqueName: opts.ScaleName,
		InitFN:     opts.ScaleInit,
	})
	bias := nn.AddBias(lt, tensor.Shape{1, opts.InputSize}, NewWeightsOpts{
		UniqueName: opts.BiasName,
		InitFN:     opts.BiasInit,
	})

	return &BatchNormModule{
		model: nn,
		layer: lt,
		opts:  opts,
		scale: scale,
		bias:  bias,
	}
}

// BatchNorm2d defines the batch norm operation for tensors with shape (B, C, W, H)
func BatchNorm2d(nn *Model, opts BatchNormOpts) *BatchNormModule {
	batchSize, _ := strconv.Atoi(os.Getenv("batchSize"))
	opts.setDefaults()
	lt := godl.AddLayer("BatchNorm2d")

	scale := nn.AddLearnable(lt, "scale", tensor.Shape{batchSize, opts.InputSize, 1, 1}, NewWeightsOpts{
		UniqueName: opts.ScaleName,
		InitFN:     opts.ScaleInit,
	})
	bias := nn.AddBias(lt, tensor.Shape{batchSize, opts.InputSize, 1, 1}, NewWeightsOpts{
		UniqueName: opts.BiasName,
		InitFN:     opts.BiasInit,
	})

	return &BatchNormModule{
		model: nn,
		layer: lt,
		opts:  opts,
		scale: scale,
		bias:  bias,
	}
}
