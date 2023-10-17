import json
import os
import numpy as np
import heapq
import random

from utils import (
    read_lines,
    get_final_prompt,
    load_cls_data,
    extract_numbers,
    k_init_pop,
)
from infer import evaluate_optimized_prompt
from llm_client import paraphrase, llm_query
from data.templates import templates
from data.template_ga import templates_2


class Evoluter:
    def __init__(self, args, evaluator):
        self.evaluator = evaluator
        self.init_poplulation = []
        self.population = []
        self.scores = []
        self.marks = []
        self.client, self.llm_config = evaluator.client, evaluator.llm_config
        self.public_out_path = self.evaluator.public_out_path

        logger = self.logger = evaluator.logger
        logger.info("=" * 50)
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)
        self.args = args

        if args.task in ["sim", "sum"]:
            self.eval_src, self.eval_tgt = evaluator.dev_src, evaluator.dev_tgt
            self.eval_src = self.eval_src[: args.sample_num]
            self.eval_tgt = [i[: args.sample_num] for i in self.eval_tgt]
        elif args.task == "qa":
            self.eval_src, self.eval_tgt = evaluator.dev_src, evaluator.dev_tgt
        else:
            self.eval_src, self.eval_tgt = load_cls_data(
                evaluator.verbalizers, args.dev_file
            )

    def sorted(self):
        best_score = 0
        total_score = 0
        with open(os.path.join(self.public_out_path, "dev_result.txt"), "w") as wf:
            self.scores, self.population, self.marks = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(self.scores, self.population, self.marks),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                )
            )
            for score, prompt, mark in zip(self.scores, self.population, self.marks):
                score_str = "\t".join([str(round(i, 4)) for i in score])
                float_score = float(score[-1])
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                wf.write(f"{mark}\t{prompt}\t{score_str}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / len(self.scores)}\n")
            wf.close()

    def init_pop(self):
        args = self.args
        evaluator = self.evaluator
        dataset = args.dataset
        prompts2mark = {}
        manual_prompt_path = f"./data/{args.task}/{dataset}/prompts.txt"
        ape_prompt_path = f"./data/{args.task}/{dataset}/prompts_auto.txt"
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

        self.evaluated_prompts = {}
        logger = self.logger
        out_path = self.public_out_path
        cur_budget = -1
        if args.initial == "all":
            cache_path = (
                args.cache_path
                if args.cache_path
                else f"./data/{args.task}/{dataset}/seed{args.seed}/prompts{model}.json"
            )
            try:
                self.evaluated_prompts = json.load(open(cache_path, "r"))
                logger.info(f"---loading prompts from {cache_path}")
                metric_index = -1
                self.evaluated_prompts = dict(
                    sorted(
                        self.evaluated_prompts.items(),
                        key=lambda item: item[1][metric_index],
                        reverse=True,
                    )
                )
                init_population = [k for k in list(self.evaluated_prompts.keys())]
            except:
                topk_population = []
                logger.info(
                    "-----evaluating initial population and paraphrasing topk---------"
                )
                for prompt in manual_pop + ape_pop:
                    eval_res = evaluator.forward(prompt, self.eval_src, self.eval_tgt)
                    scores = eval_res["scores"]
                    self.evaluated_prompts[prompt] = scores
                    topk_population.append((scores[-1], prompt))
                topk_population.sort(reverse=True, key=lambda x: x[0])

                with open(cache_path, "w") as wf:
                    self.evaluated_prompts = dict(
                        sorted(
                            self.evaluated_prompts.items(), key=lambda item: item[1][0]
                        )
                    )
                    json.dump(self.evaluated_prompts, wf)
                init_population = [i[1] for i in topk_population]
        elif args.initial == "ape":
            init_population = read_lines(ape_prompt_path)[: args.popsize]
            prompts2mark = {i: "ape" for i in init_population}
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
                self.evaluated_prompts[prompt] = [i for i in score]
                init_population.append(prompt)
            # print(init_population)
            cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
            logger.info("cur budget is {}".format(cur_budget))

        client = evaluator.client
        llm_config = evaluator.llm_config

        # test LLM
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
            logger.info("-----paraphrasing topk---------")
            para_population = paraphrase(
                client=client, sentence=k_pop, type=args.llm_type, **llm_config
            )
            for p in para_population:
                prompts2mark[p] = "para"
                score = evaluator.forward(p, self.eval_src, self.eval_tgt)["scores"]
                self.evaluated_prompts[p] = score
            init_population = k_pop + para_population
            print(init_population)
            init_population = init_population[: args.popsize]
        elif args.initial_mode in ["topk", "bottomk", "randomk"]:
            init_population = k_init_pop(
                args.initial_mode, init_population, k=args.popsize
            )

        self.population = [i for i in init_population]
        assert len(self.population) == args.popsize

        for i in self.population:
            logger.info(i)
        with open(f"{out_path}/step0_pop_para.txt", "w") as wf:
            for i in self.population:
                score_str = "\t".join(
                    [str(round(i, 4)) for i in self.evaluated_prompts[p]]
                )
                wf.write(f"{prompts2mark[i]}\t{i}\t{score_str}\n")

        self.prompts2mark = prompts2mark
        return self.evaluated_prompts, cur_budget

    def write_step(self, step, best_score, avg_score):
        with open(os.path.join(self.public_out_path, f"step{step}_pop.txt"), "w") as wf:
            for p in self.population:
                score_str = "\t".join(
                    [str(round(i, 4)) for i in self.evaluated_prompts[p]]
                )
                wf.write(self.prompts2mark[p] + "\t" + p + "\t" + score_str + "\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def evolute(self):
        raise NotImplementedError


class ParaEvoluter(Evoluter):
    def __init__(self, args, evaluator):
        super(ParaEvoluter, self).__init__(args, evaluator)

    def init_pop(self):
        args = self.args
        logger = self.logger
        init_prompt_path = f"./data/{args.task}/{args.dataset}/prompts_auto.txt"
        self.init_population = read_lines(init_prompt_path)[: args.popsize]
        self.prompts2mark = {i: "ape" for i in self.init_population}
        logger.info("initial population:")
        for i in self.init_population:
            logger.info(i)

    def evolute(self, mode):
        self.init_pop()
        args = self.args
        k = args.popsize
        logger = self.logger
        out_path = self.public_out_path
        self.evaluated_prompts = {}
        cur_budget = -1
        topk_heap = []
        best_scores, avg_scores = [], []

        if args.initial == "ckpt":
            self.init_population = []
            logger.info("cur budget is {}".format(cur_budget))
            logger.info(f"------------load from file {args.ckpt_pop}------------")
            ckpt_pop = read_lines(args.ckpt_pop)
            for line in ckpt_pop:
                try:
                    elements = line.split("\t")
                    mark, prompt = elements[0], elements[1]
                    score = elements[2:]
                except:
                    continue
                self.prompts2mark[prompt] = mark
                mean_score = float(score)
                self.evaluated_prompts[prompt] = score
                self.init_population.append(prompt)
                heapq.heappush(topk_heap, (mean_score, prompt))

                logger.info(f"{prompt}, {self.evaluated_prompts[prompt]}")
            cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])

        _ = paraphrase(
            sentence=self.init_population[0],
            client=self.client,
            type="davinci",
            **self.llm_config,
        )
        assert mode == "topk"
        # initial population evaluation
        if args.initial != "ckpt":
            for i, prompt in enumerate(self.init_population):
                res = self.evaluator.forward(prompt, self.eval_src, self.eval_tgt)
                score = res["scores"]
                self.evaluated_prompts[prompt] = score
                mean_score = score[-1]
                score_str = "\t".join([str(round(i, 4)) for i in score])
                self.logger.info(f"manual: {prompt}, {score_str}")
                heapq.heappush(topk_heap, (mean_score, prompt))

        for step in range(cur_budget + 1, args.budget):
            best_score = 0
            total_score = 0

            self.logger.info(f"step: {step}")
            self.population, self.marks, self.scores = [], [], []
            top_k = heapq.nlargest(k, topk_heap)
            new_prompts = []
            paraphrased_prompts = paraphrase(
                sentence=[i[1] for i in top_k],
                client=self.client,
                type=args.llm_type,
                temperature=0.5,
                **self.llm_config,
            )
            for i, prompt in enumerate(paraphrased_prompts):
                self.logger.info(f"step: {step}, prompt: {prompt}")
                para_res = self.evaluator.forward(prompt, self.eval_src, self.eval_tgt)
                new_score = para_res["scores"]
                new_mean_score = new_score[-1]
                new_score_str = "\t".join([str(round(i, 4)) for i in new_score])
                self.prompts2mark[prompt] = "para"
                self.logger.info(f"paraphrased: {prompt}, {new_score_str}")
                self.logger.info(
                    f"original: {top_k[i][1]}, {self.evaluated_prompts[top_k[i][1]]}"
                )
                new_prompts.append((new_mean_score, prompt))
                self.evaluated_prompts[prompt] = new_score
            for new_prompt in new_prompts:
                # heapq.heappush(topk_heap, new_prompt)
                # if len(topk_heap) > k:
                #     heapq.heappop(topk_heap)
                heapq.heappushpop(topk_heap, new_prompt)

            for _, prompt in topk_heap:
                self.population.append(prompt)
                cur_score = float(self.evaluated_prompts[prompt][-1])
                if best_score < cur_score:
                    best_score = cur_score
                total_score += cur_score
                # self.scores.append(self.evaluated_prompts[prompt])
                mark = "manual" if prompt in self.init_population else "para"
                self.marks.append(mark)
            avg_score = total_score / len(topk_heap)
            best_scores.append(best_score)
            avg_scores.append(avg_score)
            self.write_step(step, best_score, avg_score)

        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        self.logger.info(f"best_scores: {','.join(best_scores)}")
        self.logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.logger.info(f"----------testing step {step} population----------")
        best_test_score, best_test_prompt = evaluate_optimized_prompt(
            self.population[0:1],
            self.marks[0:1],
            os.path.join(out_path, f"step{step}_pop_test.txt"),
            self.evaluator,
            args,
        )


class GAEvoluter(Evoluter):
    def __init__(self, args, evaluator):
        super(GAEvoluter, self).__init__(args, evaluator)
        try:
            self.template = templates_2[args.task]
        except:
            self.template = templates_2["sim"]

    def evolute(self):
        logger = self.logger
        self.evaluated_prompts, cur_budget = self.init_pop()
        evaluator = self.evaluator
        args = self.args
        eval_src = self.eval_src
        eval_tgt = self.eval_tgt
        out_path = self.public_out_path
        template = self.template

        best_scores = []
        avg_scores = []

        cur_best_prompt, cur_best_score = max(
            self.evaluated_prompts.items(), key=lambda x: x[1][0]
        )
        cur_best_score = cur_best_score[-1]
        fitness = np.array([self.evaluated_prompts[i][0] for i in self.population])

        for step in range(cur_budget + 1, args.budget):
            total_score = 0
            best_score = 0
            fitness = np.array([self.evaluated_prompts[i][0] for i in self.population])
            new_pop = []
            if args.sel_mode == "wheel":
                wheel_idx = np.random.choice(
                    np.arange(args.popsize),
                    size=args.popsize,
                    replace=True,
                    p=fitness / fitness.sum(),
                ).tolist()  # a temp pop to select parents
                parent_pop = [self.population[i] for i in wheel_idx]
            elif args.sel_mode in ["random", "tour"]:
                parent_pop = [i for i in self.population]

            for j in range(args.popsize):
                logger.info(f"step {step}, pop {j}")
                # print(np.random.choice(np.arange(args.popsize), size=2, replace=True,
                # p=fitness/fitness.sum()).tolist())
                if args.sel_mode in ["random", "wheel"]:
                    parents = random.sample(parent_pop, 2)
                    cand_a = parents[0]
                    cand_b = parents[1]
                elif args.sel_mode == "tour":
                    group_a = random.sample(parent_pop, 2)
                    group_b = random.sample(parent_pop, 2)
                    cand_a = max(group_a, key=lambda x: self.evaluated_prompts[x][0])
                    cand_b = max(group_b, key=lambda x: self.evaluated_prompts[x][0])

                request_content = template.replace("<prompt1>", cand_a).replace(
                    "<prompt2>", cand_b
                )
                # logger.info(f"old_child: {old_prompt}, {old_score}")
                logger.info("evolution example:")
                logger.info(request_content)
                logger.info("parents:")
                logger.info(cand_a)
                logger.info(cand_b)
                child_prompt = llm_query(
                    client=self.client,
                    data=request_content,
                    type=args.llm_type,
                    task=False,
                    temperature=0.5,
                    **self.llm_config,
                )
                logger.info(f"original child prompt: {child_prompt}")
                child_prompt = get_final_prompt(child_prompt)
                logger.info(f"child prompt: {child_prompt}")

                de_eval_res = evaluator.forward(child_prompt, eval_src, eval_tgt)
                de_hypos = de_eval_res["hypos"]
                de_scores = de_eval_res["scores"]
                de_score_str = "\t".join([str(round(i, 4)) for i in de_scores])
                new_score = de_scores[-1]

                logger.info(f"new score: {de_score_str}")
                self.prompts2mark[child_prompt] = "evoluted"

                self.evaluated_prompts[child_prompt] = de_scores
                if args.ga_mode == "std":
                    selected_prompt = child_prompt
                    selected_score = new_score
                    self.population[j] = selected_prompt

                elif args.ga_mode == "topk":
                    selected_prompt = child_prompt
                    selected_score = new_score

                new_pop.append(selected_prompt)
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score
                    if best_score > cur_best_score:
                        cur_best_score = best_score

            # self.population = new_pop
            if args.ga_mode == "topk":
                double_pop = list(set(self.population + new_pop))
                double_pop = sorted(
                    double_pop,
                    key=lambda x: self.evaluated_prompts[x][-1],
                    reverse=True,
                )
                self.population = double_pop[: args.popsize]
                total_score = sum(
                    [self.evaluated_prompts[i][-1] for i in self.population]
                )
                best_score = self.evaluated_prompts[self.population[0]][-1]
            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(step, best_score, avg_score)

            if step == args.budget - 1:
                logger.info(f"----------testing step {step} self.population----------")
                pop_marks = [self.prompts2mark[i] for i in self.population]
                pop_scores = [self.evaluated_prompts[i] for i in self.population]
                self.population, pop_scores, pop_marks = (
                    list(t)
                    for t in zip(
                        *sorted(
                            zip(self.population, pop_scores, pop_marks),
                            key=lambda x: x[1][-1],
                            reverse=True,
                        )
                    )
                )

                test_prompt_num = 3
                best_score, best_prompt = evaluate_optimized_prompt(
                    self.population[:test_prompt_num],
                    pop_marks[:test_prompt_num],
                    os.path.join(out_path, f"step{step}_pop_test.txt"),
                    evaluator,
                    args,
                )
                logger.info(
                    f"----------step {step} best score: {best_score}, best prompt: {best_prompt}----------"
                )

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


class DEEvoluter(Evoluter):
    def __init__(self, args, evaluator):
        super(DEEvoluter, self).__init__(args, evaluator)
        if args.task in ["cls", "sum"]:
            self.template = templates[args.template]["sim"]
        elif args.task == "sim":
            self.template = templates[args.template]["cls"]["sst-5"]

    def evolute(self):
        logger = self.logger
        self.evaluated_prompts, cur_budget = self.init_pop()
        evaluator = self.evaluator
        args = self.args
        eval_src = self.eval_src
        eval_tgt = self.eval_tgt
        out_path = self.public_out_path
        template = self.template

        client = evaluator.client
        out_path = evaluator.public_out_path
        llm_config = evaluator.llm_config

        prompts = []
        marks = []
        scores = []
        best_scores = []
        avg_scores = []

        cur_best_prompt, cur_best_score = max(
            self.evaluated_prompts.items(), key=lambda x: x[1][0]
        )
        cur_best_score = cur_best_score[-1]
        for step in range(cur_budget + 1, args.budget):
            logger.info(f"step: {step}")
            new_pop = []
            total_score = 0
            best_score = 0
            logger.info(f"cur dev set size: {len(eval_src)}")
            preds = []
            for j in range(args.popsize):
                logger.info(f"step{step}, pop {j}")
                old_prompt = self.population[j]
                old_hypos = None
                if old_prompt not in self.evaluated_prompts:
                    eval_res = evaluator.forward(old_prompt, eval_src, eval_tgt)
                    old_hypos = eval_res["hypos"]
                    old_scores = eval_res["scores"]
                    self.evaluated_prompts[old_prompt] = old_scores
                old_scores = self.evaluated_prompts[old_prompt]
                cur_candidates = {
                    old_prompt: {
                        "score": old_scores,
                        "mark": self.prompts2mark[old_prompt],
                        "hypos": old_hypos,
                    },
                }
                logger.info(f"original: {old_prompt}")
                old_score_str = "\t".join([str(i) for i in old_scores])
                logger.info(f"old_score: {old_score_str}")

                candidates = [self.population[k] for k in range(args.popsize) if k != j]
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

                de_eval_res = evaluator.forward(de_prompt, eval_src, eval_tgt)
                de_hypos = de_eval_res["hypos"]
                de_scores = de_eval_res["scores"]
                de_score_str = "\t".join([str(round(i, 4)) for i in de_scores])

                logger.info(f"de_score: {de_score_str}")
                self.prompts2mark[de_prompt] = "evoluted"
                cur_candidates[de_prompt] = {
                    "score": de_scores,
                    "mark": self.prompts2mark[de_prompt],
                    "hypos": de_hypos,
                }
                self.evaluated_prompts[de_prompt] = de_scores

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
            self.population = new_pop

            self.write_step(step, best_score, avg_score)

            if ((step + 1) % args.write_step == 0 and args.task == "cls") or (
                step == args.budget - 1
            ):
                logger.info(f"----------testing step {step} self.population----------")
                pop_marks = [self.prompts2mark[i] for i in self.population]
                pop_scores = [self.evaluated_prompts[i] for i in self.population]
                self.population, pop_scores, pop_marks = (
                    list(t)
                    for t in zip(
                        *sorted(
                            zip(self.population, pop_scores, pop_marks),
                            key=lambda x: x[1][-1],
                            reverse=True,
                        )
                    )
                )
                test_prompt_num = 3
                best_score, best_prompt = evaluate_optimized_prompt(
                    self.population[:test_prompt_num],
                    pop_marks[:test_prompt_num],
                    os.path.join(out_path, f"step{step}_pop_test.txt"),
                    evaluator,
                    args,
                )
                logger.info(
                    f"----------step {step} best score: {best_score}, best prompt: {best_prompt}----------"
                )

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]

        self.sorted()
