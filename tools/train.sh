#!/usr/bin/env bash

export PYTHONPATH=./
CONFIG=$1
GPUS=$2
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

python3 -m torch.distributed.launch --nproc_per_node $GPUS --master_addr $ADDR --master_port $PORT \
tools/train_engine.py --config $CONFIG