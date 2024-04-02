package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	mpi "github.com/sbromberger/gompi"
	"gorgonia.org/tensor"
)

func main() {
	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)
	start := make([]float64, 400000)
	end := make([]float64, 400000)
	newComm.BcastFloat64s(start, 0)
	parallelism := mpi.WorldSize()

	if newComm.Rank() != 0 {
		InitLoad()
		streamLength, _ := strconv.Atoi(os.Getenv("streamLength"))
		imageSizeLength, _ := strconv.Atoi(os.Getenv("imageSizeLength"))
		imageSizeWidth, _ := strconv.Atoi(os.Getenv("imageSizeWidth"))
		LabelsonehotLength, _ := strconv.Atoi(os.Getenv("LabelsonehotLength"))
		TrainX := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(streamLength, 3, imageSizeLength, imageSizeWidth),
		)
		TrainY := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(streamLength, LabelsonehotLength),
		)
		for {
			for i, x := range start {
				start[i] = x + float64(rand.Intn(10))
			}
			time.Sleep(time.Second)

			LoadingLabel(labelDir)
			para := newComm.Size()
			rank := newComm.Rank()

			TrainData := LoadingTrainData(traindataDir)
			interval := len(TrainData) / (para - 1)
			starts := (rank - 1) * interval
			ends := rank * interval
			streams := interval / streamLength

			fmt.Println("Process", rank, "has totally data interval of", starts, "to", ends, "with", streams, "streams")
			for i := 0; i < streams && (starts+streamLength) <= ends; i++ {
				starttime := time.Now()
				streamEnd := starts + streamLength
				TrainXdata, TrainYdata, err := LoadingData(TrainData, starts, streamEnd, streamLength, 1, 5)
				TrainX = tensor.New(tensor.WithBacking(TrainXdata), tensor.WithShape(streamLength, 3, imageSizeLength, imageSizeWidth))
				TrainY = tensor.New(tensor.WithBacking(TrainYdata), tensor.WithShape(streamLength, LabelsonehotLength))
				X1, _ := TrainX.At(1, 1, 1, 1)
				Y1, _ := TrainY.At(1, 1)
				fmt.Println("Process", rank, " block", i, "start", starts, "end", streamEnd, "Time", time.Since(starttime), X1, Y1, err)

				starts = streamEnd
			}

			fmt.Printf("process %v now has value of %v from main process \n", newComm.Rank(), len(start))
			newComm.SendFloat64s(start, 0, newComm.Rank())
			start, _ = newComm.RecvFloat64s(0, newComm.Rank())
		}
	}

	if newComm.Rank() == 0 {
		for j := range parallelism - 1 {
			go func() {
				for {
					start, _ := newComm.RecvFloat64s(j+1, j+1)
					for i, _ := range start {
						start[i] = start[i] * end[i]
					}
					newComm.SendFloat64s(start, j+1, j+1)
				}
			}()
		}
	}
	mpi.Stop()
}
