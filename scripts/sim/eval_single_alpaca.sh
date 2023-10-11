set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=1

BUDGET=1
POPSIZE=3
NUM_OF_MANUAL=3
SEED=15

for dataset in turkcorpus
do
python infer.py \
    --seed $SEED \
    --dataset $dataset \
    --positio pre \
    --task sim \
    --batch-size 16 \
    --prompt-num 0 \
    --language_model  alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --num_of_manual $NUM_OF_MANUAL \
    --initial_mode None \
    --random_data 0 \
    --output output \
    --initial manual \
    --content "Simplify the text."
done