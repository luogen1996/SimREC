CONFIG=$1

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  dist_train.py --config $CONFIG