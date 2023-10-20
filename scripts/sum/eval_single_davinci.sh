set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

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
    --language_model  gpt \
    --llm_type davinci \
    --outputs outputs/sum/sam/eval/davinci \
    --setting default \
    --content "Summarize the text."
done