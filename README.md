# distAIGo

Distributed GPU-accelerated ResNet image classifier training in Go, using [Gorgonia](https://github.com/gorgonia/gorgonia) for neural network computation and [Open MPI](https://www.open-mpi.org/) for weight synchronization across nodes.

## Distributed Training Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                      Master (rank 0)                    │
  │  For each epoch:                                        │
  │    1. Receive weights from ALL workers                  │
  │    2. Average element-wise: W_avg = Σ W_k / N           │
  │    3. Send W_avg to ALL workers                         │
  └──┬─────────────────┬─────────────────┬─────────────────┘
     │                 │                 │
  ┌──▼──────┐     ┌────▼────┐     ┌─────▼─────┐
  │Worker 1 │     │Worker 2 │     │Worker N   │
  │(rank 1) │     │(rank 2) │ ... │(rank N)   │
  │         │     │         │     │           │
  │Train →  │     │Train →  │     │Train →    │
  │ Send W₁ │     │ Send W₂ │     │ Send Wₙ   │
  │Recv W_avg│    │Recv W_avg│    │Recv W_avg │
  │Load &    │     │Load &   │     │Load &     │
  │continue  │     │continue │     │continue   │
  └─────────┘     └─────────┘     └───────────┘
```

- **Initialization**: Rank 0 broadcasts its randomly initialized weights to all workers via `MPI_Bcast`. Workers load these into their models so everyone starts from the same state.
- **Per-epoch sync**: After each epoch, every worker sends its updated weights to the master. The master computes the element-wise arithmetic mean across all workers and sends the result back. Each worker loads the averaged weights into its model before starting the next epoch.
- **Termination**: Master runs a fixed number of epochs (read from `.env`) then exits cleanly — no deadlock.

## Supported Models

| `resnetSerie` | Architecture  | Structure         |
|---------------|---------------|-------------------|
| `1`           | ResNet-50     | `[3, 4, 6, 3]`   |
| `2`           | ResNet-101    | `[3, 4, 23, 3]`  |
| `3`           | ResNet-152    | `[3, 8, 36, 3]`  |

All variants use bottleneck residual blocks with batch normalization and default to 100 output classes.

## Prerequisites

- Go 1.22+
- Open MPI 5.0+
- CUDA 11.5+ and cuDNN 8 (optional, for GPU acceleration)
- Docker (optional, for containerized deployment)

## Configuration

Create a `.env` file in the project root:

```env
dataPath=/path/to/dataset/
resnetSerie=1
batchSize=32
epoches=10
validateEvery=1
classesNumber=100
learningrate=0.001
imageSizeLength=224
imageSizeWidth=224
```

**Dataset layout** — images must be organized in class-named subdirectories:

```
dataPath/
├── train/
│   ├── class_a/
│   │   └── *.jpg
│   └── class_b/
│       └── *.jpg
├── valid/
│   └── ...
└── label/
    └── Labels.json
```

## Build & Run

### CPU only

```bash
go build -o distaigo .
mpirun -np 4 ./distaigo
```

### With CUDA GPU acceleration

```bash
go build -tags cuda -o distaigo .
mpirun --allow-run-as-root -np 4 ./distaigo
```

### Via Docker

```bash
docker build . -t distaigo
docker run --gpus all -v /path/to/data:/data -v $(pwd):/app -w /app distaigo \
  mpirun --allow-run-as-root -np 4 ./distaigo
```

## Project Structure

```
.
├── main.go              # Entry point, MPI init, training orchestration
├── model.go             # Model struct (graph, learnables, heatmaps)
├── restnet.go           # ResNet architecture assembler (50/101/152)
├── restblock.go         # Residual block with 1×1 shortcut downsampling
├── block.go             # Standard bottleneck block (no downsampling)
├── bblock.go            # Conv → BatchNorm → ReLU → MaxPool → Dropout
├── vorblock.go          # Pre-ResNet initial block (7×7 conv)
├── identity.go          # Skip connection (1×1 conv for dimension matching)
├── conv2d.go            # 2D convolution module
├── batch_norm.go        # BatchNorm1d / BatchNorm2d
├── fc.go                # Fully connected (linear) layer
├── sequential.go        # Sequential module container
├── train.go             # TrainOpts struct & reusable Train() function
├── trainopts.go         # Hyperparameter loading from .env
├── pretrain.go          # Graph initialization and DOT file export
├── validate.go          # Validation with confusion matrix metrics
├── data.go              # Image loading from class subdirectories
├── data_loader.go       # Mini-batch data loader with shuffle
├── weights.go           # Learnable parameter node creation
├── assignment.go        # Weight serialization/deserialization for MPI
├── constants.go         # Global variables and path configuration
├── watch.go             # Debug tensor watchers
├── print.go             # Colored logging utilities
├── memstate.go          # Go runtime + CUDA memory profiling (CGo)
├── pprof.go             # CPU/memory pprof setup
├── cudamodules.go       # CUDA PTX kernel registrations (build tag: cuda)
├── maxmul.cu            # CUDA helper: GPU free memory query
├── gompi/               # MPI wrapper sub-package (gompi bindings)
├── fc/                  # Standalone FC layer sub-module
├── Dockerfile           # CUDA 11.5 + Go 1.22 + Open MPI 5.0 image
└── go.mod               # Go module definition
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `gorgonia.org/gorgonia` | Computational graph & automatic differentiation |
| `gorgonia.org/tensor` | Multi-dimensional tensor operations |
| `gorgonia.org/cu` | CUDA GPU backend |
| `github.com/sbromberger/gompi` | Open MPI bindings for Go |
| `github.com/dcu/godl` | High-level deep learning module abstractions |
| `gonum.org/v1/plot` | Weight/gradient heatmap visualization |
| `github.com/olekukonko/tablewriter` | Confusion matrix table rendering |
