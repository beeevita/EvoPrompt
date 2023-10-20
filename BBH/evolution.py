import json
import os
from tqdm import tqdm
import random
import numpy as np
import functools
from run_bbh import eval_task
from data.templates import *
from llm_client import *
from utils import *


def evolution(args, llm_config, client):
    task = args.task
    out_path = args.output
    set_seed(args.seed)

    task_data = json.load(open("data/%s.json" % task))["examples"]
    dev_data = random.sample(task_data, args.sample_num)
    test_data = [i for i in task_data if i not in dev_data]
    model = "turbo" if "turbo" in args.llm_type else "davinci"
    task_prompt = open("lib_prompt/%s.txt" % task, "r").read()

    logger = setup_log(os.path.join(out_path, f"evol.log"))
    logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
    logger.info("=" * 50)
    eval_func = functools.partial(
        eval_task,
        task=task,
        task_prompt=task_prompt,
        eval_data=dev_data,
        client=client,
        model_index=model,
        logger=logger,
        demon=args.demon,
        **llm_config,
    )

    cache_path = (
        args.cache_path
        if args.cache_path
        else f"./cache/{args.task}/seed{args.seed}/prompts{model}.json"
    )
    print(cache_path)
    cur_budget = -1
    evaluated_prompts = {}
    prompts2mark = {}

    if args.initial == "ckpt":
        init_population = []
        logger.info(f"------------load from file {args.ckpt_pop}------------")
        ckpt_pop = read_lines(args.ckpt_pop)[: args.popsize]
        for line in ckpt_pop:
            try:
                mark, prompt, score = line.strip().split("\t")
                score = float(score)
            except:
                continue
            prompts2mark[prompt] = mark
            evaluated_prompts[prompt] = score
            init_population.append(prompt)
            cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
        logger.info("current budget: %d" % cur_budget)
    else:
        try:
            evaluated_prompts = json.load(open(cache_path, "r"))
            logger.info(f"---loading prompts from {cache_path}")
            evaluated_prompts = dict(
                sorted(
                    evaluated_prompts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            init_population = [k for k in list(evaluated_prompts.keys())]
        except:
            topk_population = []
            evaluated_prompts = {}
            prompt_file = (
                f"auto_prompts/{args.task}.txt"
                if args.initial == "ape"
                else "prompts.txt"
            )
            pop = read_lines(prompt_file)[: args.popsize]
            logger.info(
                "-----evaluating initial population and paraphrasing topk---------"
            )
            for prompt in pop:
                eval_res = eval_func(cot_prompt=prompt)
                evaluated_prompts[prompt] = eval_res
                topk_population.append((eval_res, prompt))
            topk_population.sort(reverse=True, key=lambda x: x[0])

            with open(cache_path, "w") as wf:
                evaluated_prompts = dict(
                    sorted(evaluated_prompts.items(), key=lambda item: item[1])
                )
                json.dump(evaluated_prompts, wf)
            init_population = [i[1] for i in topk_population]

        prompts2mark = {i: "manual" for i in init_population}

        if args.initial_mode in ["para_topk", "para_bottomk", "para_randomk"]:
            k_pop = k_init_pop(args.initial_mode, init_population, k=args.popsize)
            para_population = paraphrase(
                client=client, sentence=k_pop, type=args.llm_type, **llm_config
            )
            for i in para_population:
                prompts2mark[i] = "para"
            init_population = k_pop + para_population
            # print(init_population)
            init_population = init_population[: args.popsize]
        elif args.initial_mode in ["topk", "bottomk", "randomk"]:
            init_population = k_init_pop(
                args.initial_mode, init_population, k=args.popsize
            )
    cur_best_score = 0
    cur_best_prompt = ""
    with open(os.path.join(out_path, "step0_pop_para.txt"), "w") as wf:
        for i in init_population:
            if i not in evaluated_prompts:
                init_scores = eval_func(cot_prompt=i)
                evaluated_prompts[i] = init_scores
            scores = evaluated_prompts[i]
            if cur_best_score < scores:
                cur_best_score = scores
                cur_best_prompt = i
            wf.write(f"{prompts2mark[i]}\t{i}\t{scores}\n")
    population = [i for i in init_population]

    template = templates[args.template]["sim"]

    prompts = []
    marks = []
    scores = []
    best_scores = []

    avg_scores = []
    for i in range(cur_budget + 1, args.budget):
        logger.info(f"step: {i}")
        new_pop = []
        total_score = 0
        best_score = 0
        preds = []
        for j in range(args.popsize):
            logger.info("step {i}, pop {j}".format(i=i, j=j))
            old_prompt = population[j]
            if old_prompt not in evaluated_prompts:
                eval_res = eval_func(cot_prompt=old_prompt)
                evaluated_prompts[old_prompt] = eval_res
            old_scores = evaluated_prompts[old_prompt]
            cur_candidates = {
                old_prompt: {
                    "score": old_scores,
                    "mark": prompts2mark[old_prompt],
                },
            }
            logger.info(f"original: {old_prompt}")
            logger.info(f"old_score: {old_scores}")

            candidates = [population[k] for k in range(args.popsize) if k != j]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            if not args.donor_random:
                c = cur_best_prompt
            request_content = (
                template.replace("<prompt0>", old_prompt)
                .replace("<prompt1>", a)
                .replace("<prompt2>", b)
                .replace("<prompt3>", c)
            )
            # if j == 0:
            logger.info("evolution example:")
            logger.info(request_content)
            logger.info("parents:")
            logger.info(a)
            logger.info(b)
            # logger.info(f"old_child: {old_prompt}, {old_score}")
            de_prompt = llm_query(
                client=client,
                data=request_content,
                type=args.llm_type,
                task=False,
                temperature=0.5,
                **llm_config,
            )
            logger.info(f"de original prompt: {de_prompt}")
            de_prompt = get_final_prompt(de_prompt)
            logger.info(f"de prompt: {de_prompt}")

            de_eval_res = eval_func(cot_prompt=de_prompt)
            logger.info(f"de_score: {de_eval_res}")
            prompts2mark[de_prompt] = "evoluted"
            cur_candidates[de_prompt] = {
                "score": de_eval_res,
                "mark": prompts2mark[de_prompt],
            }
            evaluated_prompts[de_prompt] = de_eval_res

            selected_prompt = max(
                cur_candidates, key=lambda x: cur_candidates[x]["score"]
            )
            selected_score = float(cur_candidates[selected_prompt]["score"])
            selected_mark = cur_candidates[selected_prompt]["mark"]
            total_score += selected_score
            if selected_score > best_score:
                best_score = selected_score
                if best_score > cur_best_score:
                    cur_best_score = best_score
                    cur_best_prompt = selected_prompt

            new_pop.append(selected_prompt)
            if selected_prompt not in prompts:
                prompts.append(selected_prompt)
                scores.append(selected_score)
                marks.append(selected_mark)
            logger.info("\n")

        avg_score = total_score / args.popsize
        avg_scores.append(avg_score)
        best_scores.append(best_score)
        population = new_pop

        with open(os.path.join(out_path, f"step{i}_pop.txt"), "w") as wf:
            for p in population:
                score = evaluated_prompts[p]
                wf.write(f"{prompts2mark[p]}\t{p}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

        if ((i + 1) % 10 == 0 and args.task == "cls") or (i == args.budget - 1):
            logger.info(f"----------testing step{i} population----------")
            pop_marks = [prompts2mark[i] for i in population]
            pop_scores = [evaluated_prompts[i] for i in population]
            population, pop_scores, pop_marks = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(population, pop_scores, pop_marks),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
            )

            test_prompt_num = args.popsize // 2
            with open(os.path.join(out_path, f"step{i}_pop_test.txt"), "w") as wf:
                for i in range(test_prompt_num):
                    test_prompt = population[i]
                    test_mark = pop_marks[i]
                    test_score = eval_func(cot_prompt=test_prompt, eval_data=test_data)
                    dev_score = evaluated_prompts[test_prompt]
                    all_score = (
                        test_score * len(test_data)
                        + len(dev_data) * evaluated_prompts[test_prompt]
                    ) / len(task_data)
                    wf.write(
                        f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\n"
                    )
                    wf.flush()

    best_scores = [str(i) for i in best_scores]
    avg_scores = [str(round(i, 4)) for i in avg_scores]
    logger.info(f"best_scores: {','.join(best_scores)}")
    logger.info(f"avg_scores: {','.join(avg_scores)}")
    pop_scores = [evaluated_prompts[i] for i in population]
    pop_marks = [prompts2mark[i] for i in population]
    sort_write(
        population, pop_scores, pop_marks, os.path.join(out_path, f"dev_result.txt")
    )


def sort_write(population, pop_scores, pop_marks, write_path):
    with open(write_path, "w") as wf:
        population, pop_scores, pop_marks = (
            list(t)
            for t in zip(
                *sorted(
                    zip(population, pop_scores, pop_marks),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )
        for prompt, score, mark in zip(population, pop_scores, pop_marks):
            wf.write(f"{mark}\t{prompt}\t{score}\n")
        wf.close()
