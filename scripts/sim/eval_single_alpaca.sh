set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=1


SEED=15

for dataset in asset
do
python infer.py \
    --seed $SEED \
    --dataset $dataset \
    --positio pre \
    --task sim \
    --batch-size 16 \
    --prompt-num 0 \
    --language_model  alpaca \
    --output outputs/sim/eval/alpaca \
    --content "Simplify the text."
done