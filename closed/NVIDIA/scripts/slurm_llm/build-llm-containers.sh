#!/bin/bash

trtllm_ngc_container=nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1

while [[ $# -gt 0 ]]; do
    case $1 in
        --trtllm_ngc_container)
            trtllm_ngc_container="$2"
            shift 2
            ;;
    esac
done

root=$(git rev-parse --show-toplevel)/closed/NVIDIA

mkdir -p $root/build/
rm -rf $root/build/inference && rm -rf $root/build/mitten

git clone https://github.com/mlcommons/inference.git $root/build/inference
git clone https://github.com/NVIDIA/mitten.git $root/build/mitten

srun --ntasks=1 --nodes=1 \
    --container-image=$trtllm_ngc_container \
    --container-save=./llm-mlpinf-container.sqsh \
    --container-mounts=$root:/work \
    --container-workdir=/work \
    --container-remap-root /work/scripts/slurm_llm/install_mlperf_deps.sh
