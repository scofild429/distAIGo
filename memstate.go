package main

/*
int maxmul();
#cgo LDFLAGS: -L. -L./ -lmaxmul
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"
)

func GetMemFile(startrecode, stoprecode chan string, millisecond int64) {
	<-startrecode

	records := "|seconds|mem|Heap|GC|"
	var memStats runtime.MemStats
	count := 1

	for {
		time.Sleep(time.Millisecond * time.Duration(millisecond))
		select {
		case <-stoprecode:
			h, err := os.Create(GetMemFileName)
			if err != nil {
				fmt.Println(err)
				h.Close()
				return
			}
			fmt.Fprintln(h, records)
			h.Close()
			return
		default:
			runtime.ReadMemStats(&memStats)
			new := "\n |" + strconv.Itoa(count) +
				"|" + strconv.FormatUint(memStats.Alloc/1024/1024, 10) +
				"|" + strconv.FormatUint(memStats.HeapAlloc/1024/1024, 10) +
				"|" + strconv.Itoa(int(memStats.NumGC)) + "|"
			records += new
			count += 1
		}
	}
}

func GetMemCuda(startcudarecode, stopcudarecode chan string, millisecond int64) {

	records := "|seconds|mem|"

	<-startcudarecode
	count := 1
	var mem int32
	for {
		time.Sleep(time.Millisecond * time.Duration(millisecond))
		select {
		case <-stopcudarecode:
			f, err := os.Create(GetMemCudaName)
			if err != nil {
				fmt.Println(err)
				f.Close()
				return
			}
			fmt.Fprintln(f, records)
			f.Close()
			return
		default:
			mem = int32(C.maxmul())
			new := "\n |" + strconv.Itoa(count) + "|" + strconv.Itoa(int(mem)) + "|"
			records += new
			count += 1
		}
	}
}
