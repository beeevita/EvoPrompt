# evaluating GPT-3.5 turbo model on BBH

import openai
import json
import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from utils import extract_ans, batchify
from llm_client import turbo_query, davinci_query

MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def create_dataset(mode, task_prompt, cot_prompt, eval_data,demon=1):
    questions = []
    prompt_qs = []
    answers= []
    for q_ in eval_data:
        task_prompt = task_prompt.replace('<prompt>', cot_prompt)
        if demon: 
            q = '\n\nQ: ' + q_['input']
            prompt_q = task_prompt + q + f"\nA: {cot_prompt}"
        else:
            q = 'Q: ' + q_['input']
            prompt_q = q + f"\nA: {cot_prompt}"
        questions.append(q)
        prompt_qs.append(prompt_q)
        if mode == 'multiple_choice':
            a = q_['target'][1]
        elif mode == 'free_form':
            a = q_['target']
        answers.append(a)
    return prompt_qs, questions,answers


def eval_task(task, task_prompt,cot_prompt,eval_data, client, model_index,logger,demon ,**kwargs):
    # for task in tasks:
    # print('Testing %s ...' % task)
    correct = 0
    mode = 'multiple_choice' if task in MULTIPLE_CHOICE_TASKS else 'free_form'
    print_first = True
    prompt_qs, questions,answers = create_dataset(mode, task_prompt, cot_prompt, eval_data, demon)
    if 'turbo' in model_index:
        for i in tqdm(range(len(prompt_qs))):
            prompt_q = prompt_qs[i]
            q = questions[i]
            a = answers[i]

        # for prompt_q,q,a in tqdm(zip(prompt_qs, questions,answers)):
            ans_model = turbo_query(prompt_q, temperature=0,**kwargs)
            ans_ = extract_ans(ans_model, mode)
            if print_first:
                logger.info('First prompt: ')
                logger.info(prompt_q)
                logger.info("first answer: ")
                logger.info(ans_model)
                logger.info(ans_)
                print_first = False
            
            if ans_ == a:
                correct += 1
    else:
        batched_prompt_qa = batchify(prompt_qs)
        responses= []
        for batch in tqdm(batched_prompt_qa):
            if print_first:
                logger.info('First prompt: ')
                logger.info(batch[0])
                print_first = False
            response = davinci_query(batch, client,temperature=0,**kwargs)
            responses.extend(response)
        for ans, q, a in zip(responses, questions, answers):
            ans_ = extract_ans(ans, mode)
            if ans_ == a:
                correct += 1
    accuracy = correct / len(eval_data)
    print('%s acc %.4f' % (task, correct / len(eval_data)))
    return accuracy

