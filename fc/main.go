package main

import (
	"fmt"

	mpi "github.com/sbromberger/gompi"
)

func mainm(){
	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)
	start := make([]float64, 400000)
	end := make([]float64, 400000)
	newComm.BcastFloat64s(start, 0)
	
	if newComm.Rank() != 0 {
		for {
			//			model_start(newComm.Rank())
			fmt.Printf("process %v now has value of %v from main process \n", newComm.Rank(), len(start))
			newComm.SendFloat64s(start, 0, newComm.Rank())
			start, _ = newComm.RecvFloat64s(0, newComm.Rank())
		}
	}

	if newComm.Rank() == 0 {
		go func (){
			for { 
				start, _ := newComm.RecvFloat64s(1, 1)
				for i, _ := range start {
					start[i] = start[i] * end[i]
				}
				newComm.SendFloat64s(start, 1, 1)
			}
		}()

		go func (){
			for { 
				start, _ := newComm.RecvFloat64s(2, 2)
				for i, _ := range start {
					start[i] = start[i] * end[i]
				}
				newComm.SendFloat64s(start, 2, 2)
			}
		}()

		go func (){
			for { 
				start, _ := newComm.RecvFloat64s(3, 3)
				for i, _ := range start {
					start[i] = start[i] * end[i]
				}
				newComm.SendFloat64s(start, 3, 3)
			}
		}()
	}
	mpi.Stop()
}

