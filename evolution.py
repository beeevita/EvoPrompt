import os
import random

# import nevergrad as ng
import numpy as np
import torch
import json

from utils import *
from llm_client import *
from infer import evaluate_optimized_prompt
from data.templates import templates

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def gpt_evo(args, evaluator):
    dataset = args.dataset
    prompts2mark = {}
    manual_prompt_path = f"./data/{args.task}/{dataset}/prompts_pre.txt"
    ape_prompt_path = f"./data/{args.task}/{dataset}/prompts_pre_ape.txt"
    if "gpt" in args.language_model or "opt" in args.language_model:
        model = f"_{args.language_model}"
    else:
        model = ""

    manual_pop = read_lines(manual_prompt_path)
    try:
        ape_pop = read_lines(ape_prompt_path)
    except:
        ape_pop = []
    for p in ape_pop:
        prompts2mark[p] = "ape"
    for p in manual_pop:
        prompts2mark[p] = "manual"

    if args.task == "cls":
        eval_src, eval_tgt = load_cls_data(evaluator.verbalizers, args.dev_file)
    else:
        eval_src, eval_tgt = evaluator.dev_src, evaluator.dev_tgt

    evaluated_prompts = {}
    logger = evaluator.logger
    cur_budget = -1
    if args.initial == "all":
        cache_path = (
            args.cache_path
            if args.cache_path
            else f"./cache/{args.task}/seed{args.seed}/prompts{model}.json"
        )
        try:
            evaluated_prompts = json.load(open(cache_path, "r"))
            logger.info(f"---loading prompts from {cache_path}")
            metric_index = 0 if args.metric == "bleu" else -1
            evaluated_prompts = dict(
                sorted(
                    evaluated_prompts.items(),
                    key=lambda item: item[1][metric_index],
                    reverse=True,
                )
            )
            init_population = [k for k in list(evaluated_prompts.keys())]
        except:
            topk_population = []
            logger.info(
                "-----evaluating initial population and paraphrasing topk---------"
            )
            for prompt in manual_pop + ape_pop:
                eval_res = evaluator.forward(prompt, "", eval_src, eval_tgt)
                scores = eval_res["scores"]
                evaluated_prompts[prompt] = scores
                topk_population.append((scores[-1], prompt))
            topk_population.sort(reverse=True, key=lambda x: x[0])

            with open(cache_path, "w") as wf:
                evaluated_prompts = dict(
                    sorted(evaluated_prompts.items(), key=lambda item: item[1][0])
                )
                json.dump(evaluated_prompts, wf)
            init_population = [i[1] for i in topk_population]
    elif args.initial == "ape":
        init_population = read_lines(ape_prompt_path)[: args.popsize]
        prompts2mark = {i: "ape" for i in init_population}
    elif args.initial == "manual":
        cache_path = f"./data/{args.task}/{dataset}/seed{args.seed}/prompts.json"
        evaluated_prompts = json.load(open(cache_path, "r"))
        logger.info(f"---loading prompts from {cache_path}")
        evaluated_prompts = dict(
            sorted(evaluated_prompts.items(), key=lambda item: item[1][0], reverse=True)
        )

        init_population = read_lines(manual_prompt_path)[: args.popsize]
        init_population = sorted(init_population, key=lambda x: evaluated_prompts[x][0])
        prompts2mark = {i: "manual" for i in init_population}
    elif args.initial == "ckpt":
        init_population = []
        logger.info(f"------------load from file {args.ckpt_pop}------------")
        ckpt_pop = read_lines(args.ckpt_pop)[: args.popsize]
        for line in ckpt_pop:
            try:
                elements = line.split("\t")
                mark, prompt = elements[0], elements[1]
                score = elements[2:]
                score = [float(i) for i in score]
            except:
                continue
            prompts2mark[prompt] = mark
            evaluated_prompts[prompt] = score
            init_population.append(prompt)
        # args.popsize = len(ckpt_pop)
        # print(extract_numbers(args.ckpt_pop.split('/')[-1]))
        cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
        logger.info("cur budget is {}".format(cur_budget))

    if args.task in ["cls", "sum"]:
        template = templates[args.template]["sim"]
    elif args.task == "sim":
        template = templates[args.template]["cls"]["sst-5"]

    print(template)
    client = evaluator.client
    out_path = evaluator.public_out_path
    llm_config = evaluator.llm_config

    # test LLM client
    _ = paraphrase(
        sentence="Hi, I am a student.",
        type=args.llm_type,
        client=client,
        temperature=0.5,
        **llm_config,
    )
    logger.info("test LLM client success")
    if args.initial_mode in ["para_topk", "para_bottomk", "para_randomk"]:
        k_pop = k_init_pop(args.initial_mode, init_population, k=args.popsize)
        para_population = paraphrase(
            client=client, sentence=k_pop, type=args.llm_type, **llm_config
        )
        for i in para_population:
            prompts2mark[i] = "para"
        init_population = k_pop + para_population
        init_population = init_population[: args.popsize]
    elif args.initial_mode in ["topk", "bottomk", "randomk"]:
        init_population = k_init_pop(args.initial_mode, init_population, k=args.popsize)

    population = [i for i in init_population]
    # print(population)
    # print(args.popsize)
    assert len(population) == args.popsize
    logger.info("=" * 50)
    logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
    logger.info("=" * 50)
    logger.info("initial population:")
    cur_best_score = 0
    cur_best_prompt = ""
    for i in init_population:
        logger.info(i)

    with open(os.path.join(out_path, "step0_pop_para.txt"), "w") as wf:
        for i in init_population:
            if i not in evaluated_prompts:
                init_scores = evaluator.forward(i, "", eval_src, eval_tgt, None)[
                    "scores"
                ]
                evaluated_prompts[i] = init_scores
            scores = evaluated_prompts[i]
            if cur_best_score < scores[-1]:
                cur_best_score = scores[-1]
                cur_best_prompt = i
            wf.write(f"{prompts2mark[i]}\t{i}\t{' '.join([str(j) for j in scores])}\n")

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
        logger.info(f"cur dev set size: {len(eval_src)}")
        preds = []
        for j in range(args.popsize):
            logger.info("step {i}, pop {j}".format(i=i, j=j))
            old_prompt = population[j]
            old_hypos = None
            if old_prompt not in evaluated_prompts:
                eval_res = evaluator.forward(old_prompt, "", eval_src, eval_tgt, "cot")
                old_hypos = eval_res["hypos"]
                old_scores = eval_res["scores"]
                evaluated_prompts[old_prompt] = old_scores
            old_scores = evaluated_prompts[old_prompt]
            cur_candidates = {
                old_prompt: {
                    "score": old_scores,
                    "mark": prompts2mark[old_prompt],
                    "hypos": old_hypos,
                },
            }
            logger.info(f"original: {old_prompt}")
            old_score_str = "\t".join([str(i) for i in old_scores])
            logger.info(f"old_score: {old_score_str}")

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
            evaluator.logger.info("evolution example:")
            evaluator.logger.info(request_content)
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

            de_eval_res = evaluator.forward(de_prompt, "", eval_src, eval_tgt, "cot")
            de_hypos = de_eval_res["hypos"]
            de_scores = de_eval_res["scores"]
            de_score_str = "\t".join([str(round(i, 4)) for i in de_scores])

            logger.info(f"de_score: {de_score_str}")
            prompts2mark[de_prompt] = "evoluted"
            cur_candidates[de_prompt] = {
                "score": de_scores,
                "mark": prompts2mark[de_prompt],
                "hypos": de_hypos,
            }
            evaluated_prompts[de_prompt] = de_scores

            selected_prompt = max(
                cur_candidates, key=lambda x: cur_candidates[x]["score"][-1]
            )
            selected_score = float(cur_candidates[selected_prompt]["score"][-1])
            selected_mark = cur_candidates[selected_prompt]["mark"]
            total_score += selected_score
            if selected_score > best_score:
                best_score = selected_score
                if best_score > cur_best_score:
                    cur_best_score = best_score
                    cur_best_prompt = selected_prompt

            new_pop.append(selected_prompt)
            preds.append(cur_candidates[selected_prompt]["hypos"])
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
                score_str = "\t".join([str(round(i, 4)) for i in evaluated_prompts[p]])
                wf.write(prompts2mark[p] + "\t" + p + "\t" + score_str + "\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

        if ((i + 1) % args.write_step == 0 and args.task == "cls") or (
            i == args.budget - 1
        ):
            logger.info(f"----------testing step{i} population----------")
            pop_marks = [prompts2mark[i] for i in population]
            pop_scores = [evaluated_prompts[i] for i in population]
            population, pop_scores, pop_marks = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(population, pop_scores, pop_marks),
                        key=lambda x: x[1][-1],
                        reverse=True,
                    )
                )
            )

            test_prompt_num = 1 if args.task == "mt" else 3
            best_score, best_prompt = evaluate_optimized_prompt(
                population[:test_prompt_num],
                pop_marks[:test_prompt_num],
                os.path.join(out_path, f"step{i}_pop_test.txt"),
                evaluator,
                args,
            )
            logger.info(
                f"----------step {i} best score: {best_score}, best prompt: {best_prompt}----------"
            )

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
                    key=lambda x: x[1][-1],
                    reverse=True,
                )
            )
        )
        for prompt, score, mark in zip(population, pop_scores, pop_marks):
            score_str = "\t".join([str(round(i, 4)) for i in score])
            wf.write(f"{mark}\t{prompt}\t{score_str}\n")
        wf.close()


def ape(args, evaluator):
    from evoluter import ParaEvoluter

    evoluter = ParaEvoluter(args, evaluator)
    evoluter.evolute(mode=args.para_mode)


def ga_evo(args, evaluator):
    from evoluter import GAEvoluter

    evoluter = GAEvoluter(args, evaluator)
    evoluter.evolute()


def de_evo(args, evaluator):
    from evoluter import DEEvoluter

    evoluter = DEEvoluter(args, evaluator)
    evoluter.evolute()
