package main

import "github.com/dcu/godl"

// Sequential runs the given layers one after the other
func Sequential(m *Model, modules ...godl.Module) godl.ModuleList {
	_ = godl.AddLayer("Sequential")

	list := godl.ModuleList{}
	list.Add(modules...)

	return list
}
