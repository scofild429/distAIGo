package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

var (
	datasetDir string
)

func init() {
	flag.StringVar(&datasetDir, "dataset-dir", "/home/si/data/mnist", "The dir where the dataset is located")
}

func handleErr(what string, err error) {
	if err != nil {
		log.Panicf("%s: %v", what, err)
	}
}

var weights = make([]float32, 267410)
var indexs = []int{235200, 300, 300, 300, 30000, 100, 100, 100, 1000, 10}
var weight = Weightassignment()

func main() {
	rank := 5
	trainX, trainY, err := Load(ModeTrain, datasetDir)
	handleErr("loading trainig mnist data", err)
	validateX, validateY, err := Load(ModeTrain, datasetDir)
	handleErr("loading validation mnist data", err)
	fmt.Println(trainX.Shape())
	
	for i := range weights {
		weights[i] = rand.Float32()/2
	}

	
	model := NewModel()
	options := MyModuleOpts{}
	mymodule := NewMyModule(model, options)

	indexs := []int{}
	for index, item := range(model.learnables){
		val := item.Value().Data().([]float32)
		indexs = append(indexs, len(val))
		fmt.Printf("the %v learnalbes is %v, has value length of %v \n",index, item, len(val))
	}
	fmt.Println("Print the lengthes of each weight:", indexs)
	// summe := 0
    // for _, zahl := range indexs {
    //     summe += zahl
    // }
	// fmt.Printf("The learnalbes value has totally length of %v and %v\n", summe, indexs)

	
	err = Train(model, mymodule, trainX, trainY, validateX, validateY, rank, TrainOpts{
		Epochs:           1,
		ValidateEvery:    1,
		BatchSize:        1000,
		WriteGraphFileTo: "graph.svg",
		Solver:           gorgonia.NewAdamSolver(gorgonia.WithLearnRate(5e-4)),
		CostObserver: func(epoch, totalEpoch, batch, totalBatch int, cost float32, rank int) {
			log.Printf("Rank: %d -- batch=%d/%d epoch=%d/%d cost=%0.3f", rank, batch, totalBatch, epoch, totalEpoch, cost)
		},
		MatchTypeFor: func(predVal, targetVal []float32) MatchType {
			var (
				rowLabel int
				yRowHigh float32
			)

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowLabel = 0
					yRowHigh = targetVal[k]
				} else if targetVal[k] > yRowHigh {
					rowLabel = k
					yRowHigh = targetVal[k]
				}
			}

			var (
				rowGuess    int
				predRowHigh float32
			)

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowGuess = 0
					predRowHigh = predVal[k]
				} else if predVal[k] > predRowHigh {
					rowGuess = k
					predRowHigh = predVal[k]
				}
			}

			if rowLabel == rowGuess {
				return MatchTypeTruePositive
			}

			return MatchTypeFalseNegative
		},
		ValidationObserver: func(confMat ConfusionMatrix, cost float32) {
			fmt.Printf("%v\nCost: %0.4f", confMat, cost)
		},
		CostFn: godl.CategoricalCrossEntropyLoss(godl.CrossEntropyLossOpt{}),
	})
	handleErr("training", err)
}
