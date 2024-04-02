package main

import (
	"fmt"

	"github.com/dcu/godl"
)

type MyModuleOpts struct {
}

func (o *MyModuleOpts) setDefaults() {
	fmt.Println("Set the RestNet Options... ")
}

type MyModule struct {
	model *Model
	opts  MyModuleOpts
	layer godl.LayerType
	seq   godl.ModuleList
}

func (m *MyModule) Name() string {
	return "MyModule"
}

func (m *MyModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}
	x := inputs[0]

	return m.seq.Forward(x)
}


func NewMyModule(model *Model, opts MyModuleOpts) *MyModule {
	opts.setDefaults()
	lt := godl.AddLayer("MyModule")

	layer := Sequential(
		model,
		Linear(model, LinearOpts{
			InputDimension:  784,
			OutputDimension: 300,
			WithBias:        true,
		}),
		BatchNorm1d(model, BatchNormOpts{
			InputSize: 300,
		}),
		godl.Rectify(),
		Linear(model, LinearOpts{
			InputDimension:  300,
			OutputDimension: 100,
			WithBias:        true,
		}),
		BatchNorm1d(model, BatchNormOpts{
			InputSize: 100,
		}),
		godl.Rectify(),
		Linear(model, LinearOpts{
			InputDimension:  100,
			OutputDimension: 10,
			WithBias:        true,
		}),
	)

	return &MyModule{
		model: model,
		opts:  opts,
		layer: lt,
		seq:   layer,
	}
}



