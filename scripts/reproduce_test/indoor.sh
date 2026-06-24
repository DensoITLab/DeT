#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate det
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/scannet_test_1500.py"
main_cfg_path="configs/jamma/indoor/test.py"
ckpt_path="weights/det.ckpt"  # Set this to your DeT checkpoint.
dump_dir="dump/det_indoor"
profiler_name="inference"
n_nodes=1  # Manually keep this the same as --nodes.
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # Per GPU.

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --profiler_name=${profiler_name} \
    --benchmark
