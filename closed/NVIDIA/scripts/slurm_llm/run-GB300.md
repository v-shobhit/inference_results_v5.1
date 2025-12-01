# Running GB300 v5.1
Only `deepseek-r1` and `llama3_1-405b` functional. 

## Container for GB300
It's highly recommended to use a TRTLLM release container on NGC as a starting point instead of the MLPerf release containers. For this, you can use [`build-gb300-containers.sh`](./build-gb300-containers.sh)


For deepseek-r1, use the _latest_ trtllm release container
```bash
sbatch --partition gb300-backfill --time 10:00 --nodes=1 ./build-gb300-containers.sh --trtllm_ngc_container nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4
```

For llama3.1-405b, please use `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1` - **later releases may suffer from a perf regression**
```bash
sbatch --partition gb300-backfill --time 10:00 --nodes=1 ./build-gb300-containers.sh --trtllm_ngc_container nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1
```

This will create an enroot SquashFS file named `./gb300-container.sqsh` for running LLM workloads.

## `deepseek-r1`

### Folder structure in `/home/mlperf_inference_storage` path
Your mlperf scratch space should strictly have the following tree: 
```bash
cd /home/mlperf_inference_storage

tree preprocessed_data/deepseek-r1/ models/hf_ckpnts/ models/deepseek-r1/ -L 1

preprocessed_data/deepseek-r1/
|-- input_ids_padded.npy
`-- input_lens.npy

models/hf_ckpnts/
`-- DeepSeek-R1-FP4 # cloned from https://huggingface.co/nvidia/DeepSeek-R1-NVFP4

models/deepseek-r1/
`-- deepseek-r1 -> ../hf_ckpnts/DeepSeek-R1-FP4

```

### Running

See `./run-trtllm-serve --help`, a sample command to run deepseek-r1 is:
```bash
salloc --partition gb300-perf \
  --time 02:00:00 \
  --nodes 2
```

After nodes are allocated:
```bash
./run-trtllm-serve.sh \
  --mlperf_scratch_space /path/to/mlperf_scratch_space \
  --trtllm_container_image ./gb300-container.sqsh \
  --mlperf_container_image ./gb300-container.sqsh \
  --scenario Offline \
  --mode serve \
  --run_client
```

Please check configs under `closed/NVIDIA/configs/GB300-NVL72_GB300-288GB_aarch64x${num_total_gpus}_TRT`. We have configs already for `8, 16, 32 and 72` GPUs. 

- This script launches one/multiple `trtllm-serve` instances via `trtllm-llmapi-launch` across nodes in a slurm job allocation. 
- If `--run_client` is specified, then it will launch a single task with `make run_harness`:
    - the trtllm-serve endpoints are launched in order of `scontrol show hostnames $SLURM_NODELIST`
    - Launches the servers, uses `telnet` to check if port(s) are active and then launches the harness. Please monitor `trtllm-serve` logs at `logs/run_${SLURM_JOBID}` to check server launch progress.
- else, servers wait indefinitely till manually killed/canceled. 


## Llama3.1-405B
You can use `local_node_instances` module for running 405B, similar to other systems. A sample command:

```bash
./local_node_instance/run_servers_and_harness.sh \
  --mlperf_container_image=./gb300-container.sqsh \
  --mlperf_scratch_path=/path/to/mlperf_inference_storage \
  --trt_engine_artefacts=/path/to/some/artefacts/ \
  --scenario=Offline \
  --benchmark_name=llama3.1-405b \
  --core_type=trtllm_endpoint \
  --num_instances_per_node=2 \
  --system_name=GB300-NVL72_GB300-288GB_aarch64x4_TRT \
  --trtllm_backend=torch \
  --hf_token=YOUR_HF_TOKEN

# --num_instances_per_node=2 -> since we use tp2pp1 (2 gpus per instance), and GB300 has 4 GPUs/node
# --hf_token=YOUR_HF_TOKEN   -> get a HF token from https://huggingface.co/settings/tokens
```

Note that it does:
1. engine generate (no-op for torch backend)
2. server launch
3. `AccuracyOnly` run
4. `PerformanceOnly` run

If you wish for only perf run, comment out the srun step that does accuracy run in [`./local_node_instances/run_servers_and_harness`](./local_node_instance/run_servers_and_harness.sh)
