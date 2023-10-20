#!/bin/bash

set -ex


llm=turbo

# for task in temporal_sequences disambiguation_qa tracking_shuffled_objects_three_objects penguins_in_a_table geometric_shapes snarks ruin_names tracking_shuffled_objects_seven_objects tracking_shuffled_objects_five_objects logical_deduction_three_objects hyperbaton logical_deduction_five_objects logical_deduction_seven_objects movie_recommendation salient_translation_error_detection reasoning_about_colored_objects date_understanding boolean_expressions multistep_arithmetic_two  navigate  dyck_languages  word_sorting  sports_understanding object_counting  formal_fallacies  causal_judgement  web_of_lies 
for task in date_understanding
do
OUT_PATH=outputs/$task/eval/$llm/0-shot
for seed in 10
do
mkdir -p $OUT_PATH/seed${seed}
python eval.py \
    --seed $seed \
    --task $task \
    --batch-size 20 \
    --sample_num 50 \
    --llm_type $llm \
    --setting default \
    --output $OUT_PATH/seed${seed}
done
done