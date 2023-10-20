set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

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
    --setting default \
    --output outputs/sum/sam/eval \
    --content "Summarize the text."
done