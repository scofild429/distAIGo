package main

import (
	"fmt"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type RestNetOpts struct {
	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn
	Learnable             bool
	PreTrained            bool
	structure             []int
	inputchannel          int
	intermediachannel     []int
}

func (o *RestNetOpts) setDefaults() {
	o.inputchannel = 64
	fmt.Println("Set the RestNet Options... ")
}

type RestNetModule struct {
	model *Model
	opts  RestNetOpts
	layer godl.LayerType
	seq   godl.ModuleList
}

func (m *RestNetModule) Name() string {
	return "RestNet"
}

func (m *RestNetModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}
	x := inputs[0]
	return m.seq.Forward(x)
}

func RestNet(m *Model, opts RestNetOpts) *RestNetModule {
	opts.setDefaults()
	lt := godl.AddLayer("RestNet")
	fixedWeights := false

	layers := []godl.Module{}
	vorblock := VorBlock(m, VorBlockOpts{
		InputDimension:  3,
		OutputDimension: 64,
		WithBias:        false,
		KernelSize:      tensor.Shape{7, 7},
		Pad:             []int{3, 3},
		Stride:          []int{2, 2},
		WeightsName:     "/layer0/conv1/7x7",
		ScaleName:       "/layer0/bn/gamma",
		bBiasName:       "/layer0/bn/beta",
	})
	layers = append(layers, vorblock)

	for layer, number := range opts.structure {
		stride := []int{1, 1}
		if layer != 0 {
			stride = []int{2, 2}
		}

		block_downsample := RestBlock(m, RestBlockOpts{
			InputDimension:  opts.inputchannel,
			OutputDimension: opts.intermediachannel[layer],
			Layer:           layer,
			Stride:          stride,
			Stage:           0,
		})
		layers = append(layers, block_downsample)

		opts.inputchannel = opts.intermediachannel[layer] * 4
		for stage := 0; stage < number; stage++ {
			block := Block(m, BlockOpts{
				InputDimension:  opts.inputchannel,
				OutputDimension: opts.intermediachannel[layer],
				Layer:           layer,
				Stage:           stage,
			})
			layers = append(layers, block)
		}
	}

	layers = append(layers,
		Linear(m, LinearOpts{
			InputDimension:  2048,
			OutputDimension: 100,
			WithBias:        false,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WeightsInit:     opts.WeightsInit,
			BiasInit:        opts.BiasInit,
			WeightsName:     "/fc1/fc1_W:0",
			BiasName:        "/fc1/fc1_b:0",
			FixedWeights:    fixedWeights,
		}),
	)

	seq := Sequential(m, layers...)

	return &RestNetModule{
		model: m,
		opts:  opts,
		layer: lt,
		seq:   seq,
	}
}
