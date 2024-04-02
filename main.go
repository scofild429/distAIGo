package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"mpi"

	"github.com/fatih/color"
	"github.com/joho/godotenv"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	err := godotenv.Load(".env")
	if err != nil {
		log.Fatalf("Some error occured when load configuation file . Err: %s", err)
	}
	dataPath := os.Getenv("dataPath")
	flag.StringVar(&traindataDir, "train", dataPath+"train", "The dir where the training dataset is located")
	flag.StringVar(&validdataDir, "valid", dataPath+"valid", "The dir where the validaiton dataset is located")
	flag.StringVar(&labelDir, "label", dataPath+"label/Labels.json", "The dir where the lables is located")
}

func main() {
	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)
	parallelism := mpi.WorldSize()

	pprofmonitor()

	// startrecode := make(chan string)
	// stoprecode := make(chan string)
	// go GetMemFile(startrecode, stoprecode, 1000)
	// startrecode <- "start"

	// startcudarecode := make(chan string)
	// stopcudarecode := make(chan string)
	// go GetMemCuda(startcudarecode, stopcudarecode, 1000)
	// startcudarecode <- "start"

	structure := make([]int, 0)
	switch resnetSerie, _ := strconv.Atoi(os.Getenv("resnetSerie")); resnetSerie {
	case 1:
		structure = append(structure, []int{3, 4, 6, 3}...) // for restNet50,
	case 2:
		structure = append(structure, []int{3, 4, 23, 3}...) // for resNet101
	case 3:
		structure = append(structure, []int{3, 8, 36, 3}...) // for resNet152,
	default:
		fmt.Println("resnetSerie must be set in configure file .env ")
		return
	}

	m := NewModel()
	options := RestNetOpts{
		PreTrained:        false,
		Learnable:         true,
		WithBias:          false,
		structure:         structure,
		intermediachannel: []int{64, 128, 256, 512},
	}
	resNet := RestNet(m, options)
	for _, item := range m.learnables {
		val := item.Value().Data().([]float32)
		WeightsLengthArray = append(WeightsLengthArray, len(val))
		WeightsLengthSum += len(val)
		WeightsLengthAccu = append(WeightsLengthAccu, WeightsLengthSum)
	}
	fmt.Println("WeightsLengthSum", WeightsLengthSum)
	fmt.Println("WeightsLengthArray", WeightsLengthArray, "has lenght of ", len(WeightsLengthArray))
	fmt.Println("WeightsLengthAccu", WeightsLengthAccu, "has lenght of ", len(WeightsLengthAccu))

	Weightscomposed = make([]float64, WeightsLengthSum)
	for _, item := range m.learnables {
		val := item.Value().Data().([]float32)
		val64 := []float64{}
		for i, _ := range val {
			val64 = append(val64, float64(val[i]))
		}
		Weightscomposed = append(Weightscomposed, val64...)
	}
	fmt.Println("BcastFloat32s from main process to work process!")
	newComm.BcastFloat64s(Weightscomposed, 0)

	if newComm.Rank() != 0 {
		// resNet = RestNet(m, options)

		LoadingLabel(labelDir)
		trainX, trainY, err := LoadTraining(traindataDir, "train")
		fmt.Printf("trainYis %v, trainY is %v, and %v\n", trainX.Shape(), trainY.Shape(), err)
		validX, validY, err := LoadTraining(validdataDir, "valid")
		fmt.Printf("trainYis %v, trainY is %v, and %v\n", validX.Shape(), validY.Shape(), err)

		opts := GetTrainOpts()
		err = Pretrain(m, opts)
		if err != nil {
			panic(err)
		}
		dl := NewDataLoader(trainX, trainY, DataLoaderOpts{
			BatchSize: opts.BatchSize,
			Shuffle:   false,
		})
		xShape := append(tensor.Shape{opts.BatchSize}, trainX.Shape()[1:]...)
		x := gorgonia.NewTensor(m.trainGraph, trainX.Dtype(), trainX.Shape().Dims(), gorgonia.WithShape(xShape...), gorgonia.WithName("x"))
		y := gorgonia.NewMatrix(m.trainGraph, trainY.Dtype(), gorgonia.WithShape(opts.BatchSize, trainY.Shape()[1]), gorgonia.WithName("y"))

		result := resNet.Forward(x)

		var (
			costVal gorgonia.Value
			predVal gorgonia.Value
		)

		{
			cost := opts.CostFn(result, y)

			gorgonia.Read(cost, &costVal)
			gorgonia.Read(result[0], &predVal)

			if _, err := gorgonia.Grad(cost, m.Learnables()...); err != nil {
				panic(err)
			}
		}

		validationGraph := m.trainGraph.SubgraphRoots(result[0])
		validationGraph.RemoveNode(y)

		m.evalGraph = validationGraph

		vmOpts := []gorgonia.VMOpt{
			gorgonia.BindDualValues(m.learnables...),
		}

		if opts.DevMode {
			vmOpts = append(
				vmOpts,
				gorgonia.TraceExec(),
				gorgonia.WithNaNWatch(),
				gorgonia.WithInfWatch(),
			)
		}

		vm := gorgonia.NewTapeMachine(m.trainGraph, vmOpts...)

		if opts.Solver == nil {
			info("defaulting to RMS solver")

			opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)))
		}

		defer vm.Close()

		startTime := time.Now()

		for i := 0; i < opts.Epochs; i++ {
			for dl.HasNext() {
				xVal, yVal := dl.Next()

				err := gorgonia.Let(x, xVal)
				if err != nil {
					fatal("error assigning x: %v", err)
				}

				err = gorgonia.Let(y, yVal)
				if err != nil {
					fatal("error assigning y: %v", err)
				}

				if err = vm.RunAll(); err != nil {
					fatal("Failed at epoch  %d, batch %d. Error: %v", i, dl.CurrentBatch, err)
				}

				if opts.WithLearnablesHeatmap {
					m.saveHeatmaps(i, dl.CurrentBatch, dl.opts.BatchSize, dl.FeaturesShape[0])
				}

				if err = opts.Solver.Step(gorgonia.NodesToValueGrads(m.learnables)); err != nil {
					fatal("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, dl.CurrentBatch, err)
				}

				if opts.CostObserver != nil {
					opts.CostObserver(i, opts.Epochs, dl.CurrentBatch, dl.Batches, costVal.Data().(float32))
				} else {
					color.Yellow(" Epoch %d %d | cost %v (%v)\n", i, dl.CurrentBatch, costVal, time.Since(startTime))
				}

				m.PrintWatchables()

				vm.Reset()
			}

			///////////////////////////////////////////////////////////
			for index, item := range m.learnables {
				weight := item.Value().Data().([]float32)
				weight64 := []float64{}
				for _, v := range weight {
					weight64 = append(weight64, float64(v))
				}
				end := WeightsLengthArray[index]
				if index == 0 {
					for i := 0; i < end; i++ {
						Weightscomposed[i] = weight64[i]
					}
				} else {
					for i := 0; i < end; i++ {
						Weightscomposed[WeightsLengthAccu[index-1]+i] = weight64[i]
					}
				}
			}
			/////////////////////////////////////////////////////////////
			fmt.Println("SendFloat32s: work process send to main process")
			newComm.SendFloat64s(Weightscomposed, 0, newComm.Rank())
			fmt.Println("RecvFloat32s: work process waits from main process")
			_, _ = newComm.RecvFloat64s(0, newComm.Rank())

			dl.Reset()

			if i%opts.ValidateEvery == 0 {
				err := Validate(m, x, y, costVal, predVal, validX, validY, opts)
				if err != nil {
					color.Red("Failed to run validation on epoch %v: %v", i, err)
				}
				color.Yellow(" Epoch %d | cost %v (%v)\n", i, costVal, time.Since(startTime))
			}
		}

		// res := Train(newComm, m, resNet, trainX, trainY, validX, validY, trainopts)
		// fmt.Println(res)
		// stoprecode <- "stop"
		// stopcudarecode <- "stop"
	}

	if newComm.Rank() == 0 {
		for j := range parallelism - 1 {
			go func() {
				for {
					fmt.Println("RecvFloat32s: Main process receive from work process")
					recivedWeights, _ := newComm.RecvFloat64s(j+1, j+1)
					for i, _ := range recivedWeights {
						Weightscomposed[i] += recivedWeights[i]
						Weightscomposed[i] /= 2
					}
					fmt.Println("SendFloat32s: Main send to work process")
					newComm.SendFloat64s(Weightscomposed, j+1, j+1)
				}
			}()
		}
	}
	mpi.Stop()
}
