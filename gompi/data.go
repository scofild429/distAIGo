package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/dcu/godl/imageutils"
	"github.com/hlts2/gohot"
	"github.com/joho/godotenv"
)

var (
	traindataDir string
	validdataDir string
	labelDir     string
)
var Labelsonehot = make(map[string]string)
var LabelsonehotLength = 0

func InitLoad() {
	err := godotenv.Load(".env")
	if err != nil {
		log.Fatalf("Some error occured when load configuation file . Err: %s", err)
	}
	dataPath := os.Getenv("dataPath")
	flag.StringVar(&traindataDir, "train", dataPath+"train", "The dir where the training dataset is located")
	flag.StringVar(&validdataDir, "valid", dataPath+"valid", "The dir where the validaiton dataset is located")
	flag.StringVar(&labelDir, "label", dataPath+"label/Labels.json", "The dir where the lables is located")
}

// func FuncLoadData(){
// 	InitLoad()
// 	LoadingLabel(labelDir)
// 	para := 5
// 	rank := 3
// 	streamLength, _ := strconv.Atoi(os.Getenv("streamLength"))
// 	TrainData := LoadingTrainData(traindataDir)
// 	interval := len(TrainData)/(para-1)
// 	start := (rank-1)*interval
// 	end := rank*interval
// 	streams := interval/streamLength
// 	fmt.Println("Total interval is from", start, "to", end, "with", streams, "streams")
// 	for i := 0; i<streams && (start+streamLength) <= end ; i++{
// 		streamEnd := start + streamLength
// 		LoadingData(TrainData, start, streamEnd, streamLength,  1, 5)
// 		fmt.Println(i,"th steam block start with", start, "end with", streamEnd, "train:", TrainX.Shape(), "valid:", TrainY.Shape())
// 		start = streamEnd
// 	}
// }

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
}

func LoadingTrainData(dataDir string) []map[string]string {
	dataMapArray := []map[string]string{}
	randSeed, _ := strconv.Atoi(os.Getenv("randSeed"))
	err := filepath.Walk(dataDir, func(path string, info os.FileInfo, err error) error {
		if strings.HasSuffix(path, "JPEG")  {
			s := strings.Split(path, "/")
			label := strings.Split(s[len(s)-1] , "_")[0]
			traindataMap := make(map[string]string)
			traindataMap[path]=label
			dataMapArray = append(dataMapArray, traindataMap)
		}
		return nil
	})
	if err != nil {
		fmt.Println("Load trainning data failed!")
	}
	rand.Seed(int64(randSeed))
	rand.Shuffle(len(dataMapArray), func(i, j int) {
		dataMapArray[i], dataMapArray[j] = dataMapArray[j], dataMapArray[i]
	})
	return dataMapArray
}

func LoadingData(dataMapArray []map[string]string, start int, end int, streamLength int, rank int, para int)([]float32, []float32,  error) {
	imageSizeLength, _ := strconv.Atoi(os.Getenv("imageSizeLength"))
	imageSizeWidth, _ := strconv.Atoi(os.Getenv("imageSizeWidth"))
	backingdata := []float32{}
	backinglabel := []float32{}
	loadOpts :=	imageutils.LoadOpts{
		TargetSize: []uint{uint(imageSizeLength), uint(imageSizeWidth)},
	}
	for i:= start; i<end; i++{
		for data, label := range dataMapArray[i] {
			img, err := imageutils.Load(data, loadOpts)
			if err != nil {
				fmt.Println("Train data for tensor failed!")
			}
			tensorOpts := imageutils.ToTensorOpts{
				TensorMode: "torch",
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
	}
	// TrainX := tensor.New(
	// 	tensor.Of(tensor.Float32),
	// 	tensor.WithShape(streamLength, 3, int(loadOpts.TargetSize[0]), int(loadOpts.TargetSize[1])), // count, channels, width, height
	// 	tensor.WithBacking(backingdata),
	// )
	// TrainY := tensor.New(
	// 	tensor.Of(tensor.Float32),
	// 	tensor.WithShape(streamLength, LabelsonehotLength),
	// 	tensor.WithBacking(backinglabel),
	// )
	return backingdata, backinglabel, nil
}



