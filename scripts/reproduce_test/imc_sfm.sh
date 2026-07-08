#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate det
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

python -u ./test_imc_sfm.py \
    --method det-jamma \
    --jamma_ckpt /path/to/det.ckpt \
    --subset_dir data/imc/st_peters_square/set_100/sub_set \
    --dataset_root data/imc/st_peters_square/set_100 \
    --calib_dir data/imc/st_peters_square/set_100/calibration \
    --data_cfg_path configs/data/megadepth_test_1500.py \
    --main_cfg_path configs/jamma/outdoor/test.py \
    --bag_size 5 \
    --out_json outputs/imc_sfm_det_jamma.json
