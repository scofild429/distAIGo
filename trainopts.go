package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

func GetTrainOpts() TrainOpts {

	var epoches, _ = strconv.Atoi(os.Getenv("epoches"))
	var batchSize, _ = strconv.Atoi(os.Getenv("batchSize"))
	var validateEvery, _ = strconv.Atoi(os.Getenv("validateEvery"))
	var classesNumber, _ = strconv.Atoi(os.Getenv("classesNumber"))
	var learningrate, _ = strconv.ParseFloat(os.Getenv("learningrate"), 64)

	var trainopts = TrainOpts{
		Epochs:           epoches,
		ValidateEvery:    validateEvery,
		BatchSize:        batchSize,
		WriteGraphFileTo: "",
		Solver:           gorgonia.NewAdamSolver(gorgonia.WithLearnRate(learningrate)),
		CostObserver: func(epoch, totalEpoch, batch, totalBatch int, cost float32) {
			log.Printf("batch=%d/%d epoch=%d/%d cost=%0.3f", batch, totalBatch, epoch, totalEpoch, cost)
		},
		DevMode: false,
		MatchTypeFor: func(predVal, targetVal []float32) MatchType {
			var (
				rowLabel int
				yRowHigh float32
			)
			for k := 0; k < classesNumber; k++ {
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
			for k := 0; k < classesNumber; k++ {
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
	}

	return trainopts
}
