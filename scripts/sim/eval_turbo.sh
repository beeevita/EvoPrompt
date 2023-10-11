set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

for dataset in asset
do
python infer.py \
    --dataset $dataset \
    --position pre \
    --task sim \
    --batch-size 20 \
    --prompt-num 0 \
    --language_model gpt \
    --llm_type turbo \
    --setting default \
    --initial_mode None \
    --output output/sim/$dataset/eval \
    --content "Simplify the text."
done