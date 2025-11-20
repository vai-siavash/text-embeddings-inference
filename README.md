# Text Embeddings Inference

## Internal Fork: Qwen3 Classification Support for Benchmarking

This fork adds support for **Qwen3 sequence classification models** with Flash Attention optimization for H200 GPUs. This is specifically designed for internal benchmarking and performance testing of TextGuard classification models.

### What's Different

- **Qwen3 Classification Support**: Integrated classification head support directly into `FlashQwen3Model` and `Qwen3Model`
- **Flash Attention Integration**: Optimized for H200 GPU (compute capability 90) with Flash Attention kernels
- **Docker Build Fixes**: Resolved file descriptor limits and compilation parallelism issues
- **Benchmarking Ready**: Includes `Dockerfile-cuda-bench` for streamlined benchmarking workflows

### Building for H200 (Compute Capability 90)

**Important**: This fork includes fixes for Docker build issues (ulimit, file descriptors). The build process has been optimized for stability.

```shell
# Get submodule dependencies
git submodule update --init

# Build for H200 (Hopper architecture, compute capability 90)
docker build --ulimit nofile=65536:65536 \
  -f Dockerfile-cuda \
  --build-arg CUDA_COMPUTE_CAP=90 \
  -t tei-qwen3-custom-cuda .
```

**Build Notes:**
- `--ulimit nofile=65536:65536` increases file descriptor limits to prevent "too many open files" errors
- Build time: ~20-60 minutes (depending on cache)
- The build uses single-threaded compilation (`CARGO_BUILD_JOBS=1`) for stability

### Building Benchmarking Image

For benchmarking workflows, use `Dockerfile-cuda-bench` which includes the model and pre-configured settings:

```shell
# Build benchmarking image (includes model)
docker build --ulimit nofile=65536:65536 \
  -f Dockerfile-cuda-bench \
  --build-arg CUDA_COMPUTE_CAP=90 \
  -t tei-qwen3-bench .
```

**Note**: `Dockerfile-cuda-bench` expects a `classifier/` directory with your model files in the build context.

### Running the Server

#### Standard Deployment

```shell
docker run -d \
  --gpus "device=<GPU_ID>" \
  -p 8080:80 \
  -v /path/to/classifier:/data \
  --name tei-qwen3-custom \
  tei-qwen3-custom-cuda \
  --model-id /data \
  --port 80 \
  --max-batch-tokens 32768 \
  --auto-truncate
```

#### Benchmarking Deployment

```shell
# Using the benchmarking image (model included)
docker run -d \
  --gpus "device=<GPU_ID>" \
  -p 8080:80 \
  --name tei-qwen3-bench \
  tei-qwen3-bench
```

**Key Implementation Changes:**
- Ported Qwen3 classification from upstream PR #730
- Integrated Flash Attention for H200 GPU (compute capability 90)
- Fixed Docker build issues (ulimit, file descriptors, parallelism)
- Added classification head support to `FlashQwen3Model` and `Qwen3Model`


### Troubleshooting

**Build fails with "too many open files":**
- Ensure you're using `--ulimit nofile=65536:65536` flag
- The Dockerfile already includes `ulimit -n 65536` in build commands

**CUDA errors during warmup:**
- Ensure you're using CUDA Dockerfile (`Dockerfile-cuda`)
- Verify GPU compute capability matches build arg (90 for H200)
- Check that Flash Attention is enabled (default: `USE_FLASH_ATTENTION=True`)


## Examples

- [Set up an Inference Endpoint with TEI](https://huggingface.co/learn/cookbook/automatic_embedding_tei_inference_endpoints)
- [RAG containers with TEI](https://github.com/plaggy/rag-containers)
