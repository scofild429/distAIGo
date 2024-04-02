package main
import (
	"os"
	"github.com/pkg/profile"
	"runtime/pprof"
)

func pprofmonitor(){
	cpuFile, _ := os.Create("cpu.pprof")
	pprof.StartCPUProfile(cpuFile)
	defer pprof.StopCPUProfile()
	defer profile.Start(profile.MemProfile, profile.ProfilePath(".")).Stop()
}
