import json
import os
import numpy as np
import heapq
import random
from tqdm import tqdm

from utils import setup_log, k_init_pop
from utils import (
    read_lines,
    get_final_prompt,
    extract_numbers,
)
from llm_client import paraphrase, llm_query
from data.template_ga import templates_2
from data.templates import *
from run_bbh import eval_task
import functools


class Evoluter:
    def __init__(self, args, llm_config, client):
        self.init_poplulation = []
        self.population = []
        self.scores = []
        self.marks = []
        self.prompts2mark = {}
        self.evaluated_prompts = {}

        self.client, self.llm_config = client, llm_config
        self.public_out_path = args.output
        self.task = args.task
        self.task_prompt = open("lib_prompt/%s.txt" % self.task, "r").read()

        self.logger = logger = setup_log(
            os.path.join(self.public_out_path, f"evol.log")
        )
        logger.info("=" * 50)
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)
        self.args = args

        self.out_path = os.path.join(self.public_out_path, f"dev_result.txt")
        self.task_data = json.load(open("data/%s.json" % args.task))["examples"]
        self.dev_data = random.sample(self.task_data, args.sample_num)
        self.test_data = [i for i in self.task_data if i not in self.dev_data]

        model = "turbo" if "turbo" in args.llm_type else "davinci"

        self.eval_func = functools.partial(
            eval_task,
            task=self.task,
            task_prompt=self.task_prompt,
            eval_data=self.dev_data,
            client=client,
            model_index=model,
            logger=logger,
            demon=args.demon,
            **llm_config,
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
                float_score = float(score)
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                wf.write(f"{mark}\t{prompt}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / len(self.scores)}\n")
            wf.close()

    def run(self):
        self.evolute()
        self.sorted()

    def init_pop(self):
        args = self.args
        logger = self.logger

        out_path = self.public_out_path
        cur_budget = -1
        cot_cache_path = args.cot_cache_path
        desc_cache_path = args.desc_cache_path

        def load_cache(self, cache_path):
            try:
                cache = json.load(open(cache_path, "r"))
                logger.info(f"---loading prompts from {cache_path}---")
                self.evaluated_prompts = dict(
                    sorted(
                        cache.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                init_population = [k for k in list(self.evaluated_prompts.keys())]
            except:
                topk_population = []
                self.evaluated_prompts = {}
                prompt_path = (
                    f"auto_prompts/{args.task}.txt"
                    if args.initial == "ape"
                    else "prompts.txt"
                )
                pop = read_lines(prompt_path)
                logger.info(
                    "-----evaluating initial population and paraphrasing topk---------"
                )
                for prompt in pop:
                    eval_res = self.eval_func(cot_prompt=prompt)
                    self.evaluated_prompts[prompt] = eval_res
                    topk_population.append((eval_res, prompt))
                topk_population.sort(reverse=True, key=lambda x: x[0])

                with open(cache_path, "w") as wf:
                    self.evaluated_prompts = dict(
                        sorted(self.evaluated_prompts.items(), key=lambda item: item[1])
                    )
                    json.dump(self.evaluated_prompts, wf)
                init_population = [i[1] for i in topk_population]
            return init_population, self.evaluated_prompts

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
                self.prompts2mark[prompt] = mark
                self.evaluated_prompts[prompt] = score
                init_population.append(prompt)
                cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
            logger.info("current budget: %d" % cur_budget)
        elif args.initial == "cot":
            init_population, self.evaluated_prompts = load_cache(self, cot_cache_path)
            self.prompts2mark = {i: "manual" for i in init_population}
        elif args.initial == "desc":
            init_population, self.evaluated_prompts = load_cache(self, desc_cache_path)
            self.prompts2mark = {i: "ape" for i in init_population}

        elif args.initial == "all":
            init_population_cot, self.evaluated_prompts_cot = load_cache(
                self, cot_cache_path
            )
            init_population_desc, self.evaluated_prompts_desc = load_cache(
                self, desc_cache_path
            )
            self.evaluated_prompts = {
                **self.evaluated_prompts_cot,
                **self.evaluated_prompts_desc,
            }
            self.evaluated_prompts = dict(
                sorted(
                    self.evaluated_prompts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            init_population = [k for k in list(self.evaluated_prompts.keys())]
            self.prompts2mark = {
                i: "manual" if i in init_population_cot else "ape"
                for i in init_population
            }

        # test LLM client
        _ = paraphrase(
            sentence="Hi, I am a student.",
            type=args.llm_type,
            client=self.client,
            temperature=0.5,
            **self.llm_config,
        )
        logger.info("test LLM client success")
        if args.initial_mode in ["para_topk", "para_bottomk", "para_randomk"]:
            k_pop = k_init_pop(args.initial_mode, init_population, k=args.popsize)
            para_population = paraphrase(
                client=self.client,
                sentence=k_pop,
                type=args.llm_type,
                temperature=0.5,
                **self.llm_config,
            )
            for i in para_population:
                self.prompts2mark[i] = "para"
            init_population = k_pop + para_population
            init_population = init_population[: args.popsize]
        elif args.initial_mode in ["topk", "bottomk", "randomk"]:
            init_population = k_init_pop(
                args.initial_mode, init_population, k=args.popsize
            )
        cur_best_score = 0
        cur_best_prompt = ""
        total_score = 0

        self.population = [i for i in init_population]
        assert len(self.population) == args.popsize

        with open(os.path.join(out_path, "step0_pop_para.txt"), "w") as wf:
            for i in self.population:
                if i not in self.evaluated_prompts:
                    init_scores = self.eval_func(cot_prompt=i)
                    self.evaluated_prompts[i] = init_scores
                scores = self.evaluated_prompts[i]
                total_score += scores
                if cur_best_score < scores:
                    cur_best_score = scores
                    cur_best_prompt = i
                wf.write(f"{self.prompts2mark[i]}\t{i}\t{scores}\n")
            wf.write(f"best score: {cur_best_score}\n")
            wf.write(f"average score: {total_score / args.popsize}\n")

        return self.evaluated_prompts, cur_budget

    def write_step(self, i, avg_score, best_score):
        out_path = self.public_out_path
        with open(os.path.join(out_path, f"step{i}_pop.txt"), "w") as wf:
            for p in self.population:
                score = self.evaluated_prompts[p]
                wf.write(f"{self.prompts2mark[p]}\t{p}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def evolute(self):
        raise NotImplementedError

    def test(self, step):
        self.logger.info(f"----------testing step {step} population----------")
        pop_marks = [self.prompts2mark[i] for i in self.population]
        pop_scores = [self.evaluated_prompts[i] for i in self.population]
        self.population, pop_scores, pop_marks = (
            list(t)
            for t in zip(
                *sorted(
                    zip(self.population, pop_scores, pop_marks),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )

        test_prompt_num = self.args.popsize // 2
        with open(
            os.path.join(self.public_out_path, f"step{step}_pop_test.txt"), "w"
        ) as wf:
            for i in tqdm(range(test_prompt_num)):
                test_prompt = self.population[i]
                test_mark = pop_marks[i]
                test_score = self.eval_func(
                    cot_prompt=test_prompt, eval_data=self.test_data
                )
                dev_score = self.evaluated_prompts[test_prompt]
                all_score = (
                    test_score * len(self.test_data)
                    + len(self.dev_data) * self.evaluated_prompts[test_prompt]
                ) / len(self.task_data)
                wf.write(
                    f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\n"
                )
                wf.flush()


class DEEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super(DEEvoluter, self).__init__(args, llm_config=llm_config, client=client)
        self.template = templates[args.template]["sim"]

    def evolute(self):
        logger = self.logger
        args = self.args
        self.evaluated_prompts, cur_budget = self.init_pop()
        out_path = self.public_out_path
        template = self.template
        best_scores = []
        avg_scores = []

        cur_best_prompt, cur_best_score = max(
            self.evaluated_prompts.items(), key=lambda x: x[1]
        )

        for step in range(cur_budget + 1, args.budget):
            logger.info(f"step: {step}")
            new_pop = []
            total_score = 0
            best_score = 0
            for j in range(args.popsize):
                logger.info("step {i}, pop {j}".format(i=step, j=j))
                old_prompt = self.population[j]
                if old_prompt not in self.evaluated_prompts:
                    eval_res = self.eval_func(cot_prompt=old_prompt)
                    self.evaluated_prompts[old_prompt] = eval_res
                old_scores = self.evaluated_prompts[old_prompt]
                cur_candidates = {
                    old_prompt: {
                        "score": old_scores,
                        "mark": self.prompts2mark[old_prompt],
                    },
                }
                logger.info(f"original: {old_prompt}")
                logger.info(f"old_score: {old_scores}")

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
                logger.info("evolution example:")
                logger.info(request_content)
                logger.info("parents:")
                logger.info(a)
                logger.info(b)
                # logger.info(f"old_child: {old_prompt}, {old_score}")
                de_prompt = llm_query(
                    client=self.client,
                    data=request_content,
                    type=args.llm_type,
                    task=False,
                    temperature=0.5,
                    **self.llm_config,
                )
                logger.info(f"de original prompt: {de_prompt}")
                de_prompt = get_final_prompt(de_prompt)
                logger.info(f"de prompt: {de_prompt}")

                de_eval_res = self.eval_func(cot_prompt=de_prompt)
                logger.info(f"de_score: {de_eval_res}")
                self.prompts2mark[de_prompt] = "evoluted"
                cur_candidates[de_prompt] = {
                    "score": de_eval_res,
                    "mark": self.prompts2mark[de_prompt],
                }
                self.evaluated_prompts[de_prompt] = de_eval_res

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

            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)
            self.population = new_pop

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)
            # if step == args.budget - 1:
        self.test(step=args.budget-1)

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


class GAEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super(GAEvoluter, self).__init__(args, llm_config=llm_config, client=client)
        self.template = templates_2["sim"]

    def evolute(self):
        logger = self.logger
        args = self.args
        self.evaluated_prompts, cur_budget = self.init_pop()
        out_path = self.public_out_path
        template = self.template

        best_scores = []
        avg_scores = []

        fitness = np.array([self.evaluated_prompts[i] for i in self.population])

        for step in range(cur_budget + 1, args.budget):
            total_score = 0
            best_score = 0
            fitness = np.array([self.evaluated_prompts[i] for i in self.population])
            new_pop = []
            if args.sel_mode == "wheel":
                wheel_idx = np.random.choice(
                    np.arange(args.popsize),
                    size=args.popsize,
                    replace=True,
                    p=fitness / fitness.sum(),
                ).tolist()  # temp self.population to select parents
                parent_pop = [self.population[i] for i in wheel_idx]
            elif args.sel_mode in ["random", "tour"]:
                parent_pop = [i for i in self.population]

            for j in range(args.popsize):
                logger.info("step {i}, pop {j}".format(i=step, j=j))
                if args.sel_mode in ["random", "wheel"]:
                    parents = random.sample(parent_pop, 2)
                    cand_a = parents[0]
                    cand_b = parents[1]
                elif args.sel_mode == "tour":
                    group_a = random.sample(parent_pop, 2)
                    group_b = random.sample(parent_pop, 2)
                    cand_a = max(group_a, key=lambda x: self.evaluated_prompts[x])
                    cand_b = max(group_b, key=lambda x: self.evaluated_prompts[x])

                request_content = template.replace("<prompt1>", cand_a).replace(
                    "<prompt2>", cand_b
                )
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

                de_eval_res = self.eval_func(cot_prompt=child_prompt)
                logger.info(f"new score: {de_eval_res}")
                self.prompts2mark[child_prompt] = "evoluted"

                self.evaluated_prompts[child_prompt] = de_eval_res
                if args.ga_mode == "std":
                    selected_prompt = child_prompt
                    selected_score = de_eval_res
                    self.population[j] = selected_prompt

                elif args.ga_mode == "topk":
                    selected_prompt = child_prompt
                    selected_score = de_eval_res

                new_pop.append(selected_prompt)
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score

            # self.population = new_pop
            if args.ga_mode == "topk":
                double_pop = list(set(self.population + new_pop))
                double_pop = sorted(
                    double_pop, key=lambda x: self.evaluated_prompts[x], reverse=True
                )
                self.population = double_pop[: args.popsize]
                total_score = sum([self.evaluated_prompts[i] for i in self.population])
                best_score = self.evaluated_prompts[self.population[0]]
            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)

            if step == args.budget - 1:
                self.test(step=step)

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


class ParaEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super(ParaEvoluter, self).__init__(args, llm_config=llm_config, client=client)

    def init_pop(self):
        args = self.args
        logger = self.logger
        task = args.task
        init_prompt_path = f"./auto_prompts/{task}.txt"
        self.init_population = read_lines(init_prompt_path)[: args.popsize]
        self.prompts2mark = {i: "ape" for i in self.init_population}
        logger.info("initial population:")
        for i in self.init_population:
            logger.info(i)
        with open(f"{self.public_out_path}/init.txt", "w") as wf:
            for i in self.population:
                logger.info(i)
                wf.write(f"{i}\n")

    def evolute(self):
        self.init_pop()
        args = self.args
        k = args.popsize
        logger = self.logger
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
        # initial population evaluation
        if args.initial != "ckpt":
            for i, prompt in enumerate(self.init_population):
                score = self.eval_func(cot_prompt=prompt)
                self.evaluated_prompts[prompt] = score
                self.logger.info(f"{self.prompts2mark[prompt]}: {prompt}, {score}")
                heapq.heappush(topk_heap, (score, prompt))

        for step in range(cur_budget + 1, args.budget):
            best_score = 0
            total_score = 0
            self.population, self.marks, self.scores = [], [], []
            self.logger.info(f"step: {step}")
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
                new_score = self.eval_func(cot_prompt=prompt)
                self.prompts2mark[prompt] = "para"
                self.logger.info(f"paraphrased: {prompt}, {new_score}")
                self.logger.info(
                    f"original: {top_k[i][1]}, {self.evaluated_prompts[top_k[i][1]]}"
                )
                new_prompts.append((new_score, prompt))
                self.evaluated_prompts[prompt] = new_score
            for new_prompt in new_prompts:
                heapq.heappushpop(topk_heap, new_prompt)

            for _, prompt in topk_heap:
                self.population.append(prompt)
                cur_score = float(self.evaluated_prompts[prompt])
                if best_score < cur_score:
                    best_score = cur_score
                total_score += cur_score
                mark = "manual" if prompt in self.init_population else "para"
                self.marks.append(mark)
            avg_score = total_score / len(topk_heap)
            best_scores.append(best_score)
            avg_scores.append(avg_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)

            if step == args.budget - 1:
                self.test(step=step)

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()
