GPU_ID=$1

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
DATASET="refcoco+"
SPLITBY="unc"
ID=coco+_erase

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --id ${ID} \
    --learning_rate 4e-4 \
    --learning_rate_decay_start 6 \
    --learning_rate_decay_every 6 \
    --max_epochs 30 \
    --erase_size_visual 2 \
    --erase_lang_weight 1 \
    --erase_allvisual_weight 1 \
    --erase_train 1 \
    --batch_size 20 \
    --start_from coco+_pretrain \
    2>&1 | tee logs/${ID}
