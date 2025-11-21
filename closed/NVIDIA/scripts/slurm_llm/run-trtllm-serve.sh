#!/bin/bash

#SBATCH --output=logs/run_%j/stdout.txt

# Ds-r1 constant numbers
num_nodes_dp1=2
num_tasks_per_node=4

#SBATCH --output=logs/run_%j/stdout.txt
num_nodes=$SLURM_JOB_NUM_NODES
num_total_gpus=$((num_nodes * num_tasks_per_node))

num_servers=$((num_nodes / num_nodes_dp1))

# make a temp directory ./run_xxxx
dir_name="logs/run_$SLURM_JOB_ID"
mkdir -p ./${dir_name}

actual_workdir=$(pwd)

### Flags and their defaults

# Containers
trtllm_container_image=""
mlperf_container_image=""

# NSYS flags
nsys=""
nsys_extra_flags="--cuda-event-trace=false"
profile_iter_range="3000-3200"
nsys_name="ds_r1_fp4_iters_${profile_iter_range}_%q{SLURM_NODEID}_%q{SLURM_PROCID}"

# MLPerf 
scenario="Offline"


enable_adp=1
num_requests=24000
repo_root=$(git rev-parse --show-toplevel)
dummy_weights=0
concurrency=10240
mode="bench"
run_client=0


# Batch size, MNT defaults
max_batch_size=1536
max_num_tokens=6556

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trtllm_container_image)
            trtllm_container_image="$2"
            shift 2
            ;;
        --mlperf_container_image)
            mlperf_container_image="$2"
            shift 2
            ;;
       --nsys_extra_flags)
            nsys_extra_flags="$2"
            shift 2
            ;;
        --profile_iter_range)
            profile_iter_range="$2"
            shift 2
            ;;
        --nsys_name)
            nsys_name="$2"
            shift 2
            ;;
        --nsys_prefix)
            nsys_prefix="$2"
            shift 2
            ;;
        --nsys)
            nsys="$2"
            shift 2
            ;;
        --num_requests)
            num_requests="$2"
            shift 2
            ;;
        --dummy_weights)
            dummy_weights=1
            shift
            ;;
        --concurrency)
            concurrency="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --max_batch_size)
            max_batch_size="$2"
            shift 2
            ;;
        --max_num_tokens)
            max_num_tokens="$2"
            shift 2
            ;;
        --run_client)
            run_client=1
            shift
            ;;
        --scenario)
            scenario="$2"
            shift 2
            ;;
        --mlperf_scratch_space)
            mlperf_scratch_space="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode MODE               Set mode trtllm-(bench/serve)"
            echo "  --trtllm_container_image IMAGE  Set trtllm container image"
            echo "  --mlperf_container_image IMAGE  Set mlperf container image"
            echo "  --scenario Offline|Server"
            echo "  --num_requests NUM_REQUESTS  Set number of requests (trtllm-bench only)"
            echo "  --dummy_weights           Use dummy weights (trtllm-bench only)"
            echo "  --concurrency NUM         Set concurrency level (trtllm-bench only)"
            echo "  --run_client         To run mlperf harness (trtllm-serve only)"
            echo "  --nsys PATH            Enable nsys profiling with path, profiling skipped if not set"
            echo "  --nsys_extra_flags FLAGS  Set extra nsys flags"
            echo "  --profile_iter_range RANGE              Set iteration range (default: 3000-3200)"
            echo "  --nsys_name NAME          Set nsys output name"
            echo "  --nsys_prefix PREFIX      Set nsys prefix command"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Will launch $num_servers $mode(s) on $num_nodes nodes"

# Build the command with optional nsys prefix
base_cmd="trtllm-$mode"
nsys_prefix_cmd=""
if [ -n "$nsys" ]; then
    echo "Enabling nsys profiling"
    # Export the environment variable and build the nsys command
    export TLLM_PROFILE_START_STOP="${profile_iter_range}"
    nsys_prefix_cmd="${nsys} profile --output=${dir_name}/${nsys_name} --force-overwrite=true -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop-shutdown ${nsys_extra_flags}"
fi

if [ -z "$trtllm_container_image" ]; then
    echo "trtllm container image is not set"
    exit 1
fi

if [ -z "$mlperf_scratch_space" ]; then
    echo "please set --mlperf_scratch_space"
    exit 1
fi

srun_header="srun --nodes=${num_nodes_dp1} --ntasks-per-node=${num_tasks_per_node} \
    --container-image=${trtllm_container_image} \
    --container-mounts=$(pwd)/${dir_name}:/code/tensorrt_llm/${dir_name},${mlperf_scratch_space}:/home/mlperf_inference_storage \
    --export=TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1 \
    --container-workdir=/code/tensorrt_llm \
    --mpi=pmi2 \
    --no-container-remap-root"

if [ -n "$nsys" ]; then
    srun_header="${srun_header} --export=TLLM_PROFILE_START_STOP=${profile_iter_range}"
fi

## Generate yml file. Base config is below
cat <<EOF > ${dir_name}/extra-llm-api-config.yml
enable_attention_dp: true
enable_layerwise_nvtx_marker: false
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 16, 32, 64, 128, 256, 512, 896, 1024, 1280, 1536]
disable_overlap_scheduler: false
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
moe_config:
  backend: CUTLASS
EOF

if [ "$dummy_weights" = 1 ]; then
    echo "Using dummy weights"
    echo "load_format: DUMMY" >> ${dir_name}/extra-llm-api-config.yml
fi

if [ "$enable_adp" = 1 ]; then
    echo "Enable ADP"
    echo "attention_dp_config:
  enable_balance: true
  batching_wait_iters: 2
  timeout_iters: 6" >> ${dir_name}/extra-llm-api-config.yml
fi

if [ "$mode" = "bench" ]; then 
    trtllm_flags="--model=nvidia/DeepSeek-R1-FP4 \
        --model_path=/home/mlperf_inference_storage/models/hf_ckpnts/DeepSeek-R1-FP4 \
        throughput \
        --backend pytorch \
        --extra_llm_api_options ${dir_name}/extra-llm-api-config.yml \
        --warmup 0 \
        --dataset /home/mlperf_inference_storage/preprocessed_data/trtllm-bench-datasets/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl.60N.shuffled.maxOSL_32K.txt \
        --kv_cache_free_gpu_mem_fraction 0.95 \
        --eos_id 1 \
        --tp=8 \
        --ep=8 \
        --concurrency $concurrency \
        --max_batch_size ${max_batch_size} \
        --max_num_tokens ${max_num_tokens} \
        --scheduler_policy=max_utilization \
        --num_requests $num_requests \
        --output_json ${dir_name}/outputs.json \
        --report_json ${dir_name}/report.json \
        --iteration_log ${dir_name}/iteration_log.txt"

else
    trtllm_flags="/home/mlperf_inference_storage/models/hf_ckpnts/DeepSeek-R1-FP4 \
        --host 0.0.0.0 \
        --port 30000 \
        --extra_llm_api_options ${dir_name}/extra-llm-api-config.yml \
        --num_postprocess_workers 4 \
        --tp_size 8 \
        --pp_size 1 \
        --ep_size 8 \
        --max_num_tokens ${max_num_tokens} \
        --max_batch_size ${max_batch_size} \
        --max_seq_len 23140 \
        --max_beam_width 1 \
        --tokenizer /home/mlperf_inference_storage/models/hf_ckpnts/DeepSeek-R1-FP4 \
        --backend pytorch"

fi

set -x 

# Execute the servers/benchmarks
for i in $(seq 1 $num_servers); do
    ${srun_header} \
        --container-name=trtllm-serve-container-$i \
        --output=${dir_name}/trtllm-serve-$i.txt \
        ${nsys_prefix_cmd} trtllm-llmapi-launch ${base_cmd} ${trtllm_flags} &
done

format_hostnames() {
    local num_servers=$1
    shift
    local hosts=("$@")
    local result=""

    for ((i=0; i<num_servers; i++)); do
        if [[ -n "$result" ]]; then
            result+=","
        fi
        result+="${hosts[$((i*num_nodes_dp1))]}:30000"
    done

    echo "$result"
}

node_list=$(scontrol show hostnames $SLURM_NODELIST)
endpoints=$(format_hostnames $num_servers $node_list)
echo "trtllm_server_urls: $endpoints"

if [ "$run_client" = 1 ]; then
    # warmup and health check is disabled, hence wait for server
    # Extract hostnames from endpoints and wait for port 30000 to be active on each
    TIMEOUT=1200  # 20 minutes timeout
    IFS=',' read -ra endpoint_array <<< "$endpoints"
    pids=()
    
    for endpoint in "${endpoint_array[@]}"; do
        hostname=$(echo "$endpoint" | cut -d':' -f1)
        echo "Starting port check for $hostname:30000 (timeout: ${TIMEOUT}s)"
        
        # Run overlapping job step on each node to check for port 30000
        srun --overlap --nodes=1 --nodelist="$hostname" --ntasks=1 \
            bash -c "
                TIMEOUT=$TIMEOUT
                elapsed=0
                while [ \$elapsed -lt \$TIMEOUT ]; do
                    if netstat -tuln 2>/dev/null | grep -q ':30000 .*LISTEN' || \
                       ss -tuln 2>/dev/null | grep -q ':30000 .*LISTEN'; then
                        echo \"Port 30000 is active on \$(hostname) after \${elapsed}s\"
                        exit 0
                    fi
                    sleep 2
                    elapsed=\$((elapsed + 2))
                done
                echo \"ERROR: Timeout waiting for port 30000 on \$(hostname) after \${TIMEOUT}s\"
                exit 1
            " &
        pids+=($!)
    done
    
    # Wait for all port checks to complete and check for failures
    echo "Waiting for port 30000 to be active on all nodes (timeout: ${TIMEOUT}s)..."
    all_ok=1
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            all_ok=0
        fi
    done
    
    if [ "$all_ok" -eq 0 ]; then
        echo "ERROR: One or more servers failed to start within timeout period"
        exit 1
    fi
    
    echo "All servers are ready!"
    export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=$scenario --trtllm_server_urls=${endpoints} --trtllm_runtime_flags=max_concurrency:$concurrency"
    export SYSTEM_NAME="GB300-NVL${num_total_gpus}"
    srun --overlap --nodes=1 --ntasks=1 \
        --container-image=${mlperf_container_image} \
        --mpi=pmi2 \
        --container-mounts=$(pwd):/work,${mlperf_scratch_space}:/home/mlperf_inference_storage \
        --export=RUN_ARGS,SYSTEM_NAME \
        --container-workdir=/work \
        --output=${dir_name}/mlperf-harness-run.out \
        /bin/bash -c "pip install build/inference/loadgen/mlcommons_loadgen*.whl orjson && pip install -r docker/common/requirements/requirements.llm.txt && make run_harness"

    srun --overlap --ntasks-per-node=1 \
        --container-name=trtllm-serve-container \
        pkill -9 ${base_cmd}
else
    wait
fi
