package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/dcu/godl/imageutils"
	"github.com/hlts2/gohot"
	"gorgonia.org/tensor"
)

var Labelsonehot = make(map[string]string)
var LabelsonehotLength = 0

func Getkeys[K comparable, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func LoadingLabel(labelDir string) {
	jsonFile, err := os.Open(labelDir)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened lablel file!")
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)
	labels := make(map[string]string)

	json.Unmarshal(byteValue, &labels)
	fmt.Println("Successfully Readed lablel file!")

	tokens := Getkeys(labels)
	Labelsonehot = gohot.CreateOneHotVectorFromTokens(tokens)

	for _, value := range Labelsonehot {
		LabelsonehotLength = len(value)
		break
	}
	return
}

func LoadTraining(traindataDir string, tag string) (tensor.Tensor, tensor.Tensor, error) {
	imageSizeLength, _ := strconv.Atoi(os.Getenv("imageSizeLength"))
	imageSizeWidth, _ := strconv.Atoi(os.Getenv("imageSizeWidth"))

	dataTensor, lableTensor, err := ReadAllTrainingData(
		traindataDir,
		tag,
		imageutils.LoadOpts{
			TargetSize: []uint{uint(imageSizeLength), uint(imageSizeWidth)},
		},
		imageutils.ToTensorOpts{
			TensorMode: "torch",
		},
	)
	if err != nil {
		fmt.Println(err)
	}

	return dataTensor, lableTensor, err
}
func ReadAllTrainingData(foldername string, tag string, loadOpts imageutils.LoadOpts, tensorOpts imageutils.ToTensorOpts) (tensor.Tensor, tensor.Tensor, error) {
	if len(loadOpts.TargetSize) != 2 {
		return tensor.New(tensor.WithBacking([]int{})), tensor.New(tensor.WithBacking([]int{})), fmt.Errorf("TargetSize must be defined")
	}
	datamap := make(map[string]string)
	imagesCount := 0
	backingdata := []float32{}
	backinglabel := []float32{}
	usedistence, _ := strconv.Atoi(os.Getenv("usedistence"))

	err := filepath.Walk(foldername, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println(err)
			return err
		}
		switch tag {
		case "valid":
			{
				if strings.HasSuffix(path, "JPEG") && (imagesCount%usedistence == 0) {
					label := strings.Split(path, "/")[6]
					datamap[path] = label
				}
			}
		case "train":
			{
				if strings.HasSuffix(path, "JPEG") && (imagesCount%usedistence == 0) {
					label := strings.Split(path, "/")[6]
					datamap[path] = label
				}
			}
		default:
			return nil
		}
		imagesCount++
		return nil
	})

	if err != nil {
		return tensor.New(tensor.WithBacking([]int{})), tensor.New(tensor.WithBacking([]int{})), fmt.Errorf("Open data failed")
	}

	// batchSize, _ := strconv.Atoi(os.Getenv("batchSize"))
	// datamapLength := (len(datamap) / batchSize) * batchSize

	//	datamapp := shuffleMap(datamap, datamapLength)
	//	fmt.Printf("Batchsize: %v, The data orginal length: %v, aligned data lenght: %v\n", batchSize, len(datamap), len(datamapp))

	for data, label := range datamap {
		img, err := imageutils.Load(data, loadOpts)
		if err != nil {
			return tensor.New(tensor.WithBacking([]int{})), tensor.New(tensor.WithBacking([]int{})), fmt.Errorf("Open data failed")
		}
		weightsdata := imageutils.ToArray(img, tensorOpts)
		backingdata = append(backingdata, weightsdata...)

		for _, char := range Labelsonehot[label] {
			intValue, err := strconv.Atoi(string(char))
			if err != nil {
				fmt.Println("Fehler bei der Konvertierung:", err)
			}
			backinglabel = append(backinglabel, float32(intValue))
		}
	}

	XTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(len(datamap), 3, int(loadOpts.TargetSize[0]), int(loadOpts.TargetSize[1])), // count, channels, width, height
		tensor.WithBacking(backingdata),
	)

	YTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(len(datamap), LabelsonehotLength),
		tensor.WithBacking(backinglabel),
	)
	return XTensor, YTensor, nil
}

// align the data with batchsize and shuffle
func shuffleMap(inputMap map[string]string, datamapLength int) map[string]string {
	rand.Seed(time.Now().UnixNano())

	keys := make([]string, 0, len(inputMap))
	for key := range inputMap {
		keys = append(keys, key)
	}

	rand.Shuffle(len(keys), func(i, j int) {
		keys[i], keys[j] = keys[j], keys[i]
	})

	shuffledMap := make(map[string]string)

	for i := 0; i < datamapLength; i++ {
		shuffledMap[keys[i]] = inputMap[keys[i]]
	}

	// for _, key := range keys {
	// 	shuffledMap[key] = inputMap[key]
	// }

	return shuffledMap
}
