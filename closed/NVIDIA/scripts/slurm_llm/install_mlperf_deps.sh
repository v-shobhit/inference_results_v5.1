#!/bin/bash

set -x

pip install -r /work/docker/common/requirements.aarch64-Grace.txt
pip install -r /work/docker/common/requirements/requirements.llm.txt
pip install /work/build/mitten
pip install /work/build/inference/loadgen

apt update && apt install net-tools
