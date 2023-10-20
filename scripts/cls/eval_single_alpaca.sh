set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

SEED=5

for dataset in sst2
do
python infer.py \
    --dataset $dataset \
    --seed $SEED \
    --position demon \
    --task cls \
    --batch-size 32 \
    --prompt-num 0 \
    --language_model  alpaca \
    --output outputs/cls/$dataset/eval/alpaca \
    --content "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text."
done