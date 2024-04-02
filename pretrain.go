package main

import (
	"io/ioutil"
	"log"
	"os"
)

func Pretrain(m *Model, opts TrainOpts) error {
	opts.setDefaults()

	if opts.DevMode {
		warn("Start training in dev mode")

		defer func() {
			if err := recover(); err != nil {
				graphFileName := "graph.dot"

				log.Printf("panic triggered, dumping the model graph to: %v", graphFileName)
				_ = ioutil.WriteFile(graphFileName, []byte(m.trainGraph.ToDot()), 0644)
				panic(err)
			}
		}()
	}

	if opts.WithLearnablesHeatmap {
		warn("Heatmaps will be stored in: %s", heatmapPath)
		_ = os.RemoveAll(heatmapPath)
	}

	if opts.WriteGraphFileTo != "" {
		m.WriteSVG(opts.WriteGraphFileTo)
	}

	return nil
}
