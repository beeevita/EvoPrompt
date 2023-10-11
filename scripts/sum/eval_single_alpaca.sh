set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=300
POPSIZE=30
NUM_OF_MANUAL=15
SEED=5

for dataset in sam
do
python infer.py \
    --seed $SEED \
    --dataset $dataset \
    --positio pre \
    --task sum \
    --batch-size 20 \
    --prompt-num 0 \
    --language_model alpaca \
    --random_data 0 \
    --initial manual \
    --llm_type davinci \
    --random_data 0 \
    --initial_mode None \
    --setting mt \
    --output outputs/sum/sam/gpt/eval
done