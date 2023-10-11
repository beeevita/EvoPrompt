#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

BUDGET=10
POPSIZE=10
NUM_OF_MANUAL=10
SEED=5
GA=topk
TEMPLATE=v1

for dataset in sam
do
OUT_PATH=outputs/sum/$dataset/gpt/all/ga/bd${BUDGET}_top${NUM_OF_MANUAL}_para_topk_init/$GA/$TEMPLATE-0905-fix/davinci
for SEED in 5
do
python run.py \
    --seed $SEED \
    --do_test \
    --dataset $dataset \
    --task sum \
    --batch-size 20 \
    --prompt-num 0 \
    --sample_num 100 \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --num_of_manual $NUM_OF_MANUAL \
    --random_data 0 \
    --position pre \
    --evo_mode ga \
    --llm_type davinci \
    --setting mt \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --client \
    --cache_path data/sum/$dataset/seed$SEED/prompts_gpt_tldr.json \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done
done