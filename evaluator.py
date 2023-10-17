import json
import os
from tqdm import tqdm
import numpy as np
import random
import sys
import time
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from datasets import Dataset as Dataset2
from sacrebleu.metrics import BLEU, CHRF, TER

from utils import *
from dataset import TextDataset
from llm_client import *
from metrics import *

class Evaluator(object):
    def __init__(self, args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = args.dataset
        template_file = "./data/template_v2.json"
        
        templates = json.load(open(template_file, "r"))
        if "alpaca" in args.language_model:
            model = "alpaca"
        elif "gpt" in args.language_model:
            model = "gpt"

        self.instruction_placeholder = templates["instruction"][model]
        dataset = self.dataset
        if args.position in ["icl", "pre"]:
            self.template = templates[args.task]["icl"][model][dataset][0]

        elif args.position == "demon":
            self.template = templates[args.task]["icl"][model][dataset][1]
        else:
            self.template = None
        print(self.template)
        self.model_name = args.language_model.split("/")[-1]

        self.client = None
        self.llm_config = llm_init(f"./auth.yaml", args.llm_type, args.setting)

        if "gpt" in args.language_model:
            self.tokenizer = None
            self.model = None

        elif "alpaca" in args.language_model:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'chavinlo/alpaca-native',
                # use_fast=False,
                padding_side="left",
                # truncation_side="left"
            )
            self.model = LlamaForCausalLM.from_pretrained(
                'chavinlo/alpaca-native',
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            # self.model.config.bos_token_id = 1
            # self.model.config.eos_token_id = 2
            if torch.__version__ >= "2" and sys.platform != "win32":
                self.model = torch.compile(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.language_model, torch_dtype=torch.float16
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.language_model, padding_side="left", use_fast=False
            )

        self.public_out_path = args.output
        if not os.path.exists(self.public_out_path):
            os.makedirs(self.public_out_path)
        self.logger = setup_log(os.path.join(self.public_out_path, f"evol.log"))
        logger = self.logger
        logger.info("=" * 50)
        logger.info(f"dev data: {args.dev_file}")
        logger.info(f"test data: {args.test_file}")
        self.args = args
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)
        self.logger.info(f"Instruction placeholder: {self.instruction_placeholder}")

    def form_demons(self, task, data_store_path, prompt_num):
        if task == "cls":
            data_store = read_lines(data_store_path)
            datastore_src = [line.split("\t")[0] for line in data_store]
            datastore_tgt = [
                self.verbalizers[int(line.strip().split("\t")[1])]
                for line in data_store
            ]
            demon_src, demon_tgt = extract_n_samples_per_class(
                datastore_src, datastore_tgt, prompt_num, self.dataset
            )
        elif task in ["sim", "sum"]:
            datastore_src, datastore_tgt = self.dev_src, self.dev_tgt

            indices = list(range(prompt_num))
            demon_src, demon_tgt = [datastore_src[i] for i in indices], [
                datastore_tgt[i] for i in indices
            ]
        else:
            raise ValueError("task should be sim, sum or cls")
        demonstrations = []
        for x, y in zip(demon_src, demon_tgt):
            demonstrations.append(
                format_template(
                    src=x,
                    tgt=y,
                    template=self.template,
                )
            )
        demonstrations = "\n\n".join(demonstrations)
        return demonstrations

    def create_dataset(
        self,
        data_store_path,
        test_src_sample,
        test_tgt_sample,
        tokenizer,
        verbose=True,
        src_name="",
        tgt_name="",
        model="gpt",
        batch_size=16,
        prompt_num=0,
        prompt_pre="",
        task="",
        position="pre",
    ):
        if prompt_num > 0:
            demonstrations = (
                self.form_demons(task, data_store_path, prompt_num) + "\n\n"
            )
        else:
            demonstrations = ""
        data_with_prompt = []

        if model == "gpt" and "turbo" in self.args.llm_type:
            if "turbo" in self.args.llm_type:
                data_with_prompt = test_src_sample
        else:  # davinci
            for test_src_line in test_src_sample:
                prompts = []
                example = format_template(
                    src=test_src_line,
                    src_name=src_name,
                    tgt_name=tgt_name,
                    template=self.template,
                )
                instruction_part = self.instruction_placeholder.replace(
                    "<prompt>", prompt_pre
                )

                if position in ["pre", "demon"]:  # demon includes instruction + demon
                    if "alpaca" in self.args.language_model:
                        prompts.append(instruction_part + "\n\n" + example)
                    else:
                        prompts.append(
                            instruction_part + "\n" + demonstrations + example
                        )

                elif position == "icl":  # no instruction
                    example = instruction_part + "\n" + demonstrations + example
                    prompts.append(example)
                data_with_prompt.append("\n\n".join(prompts))
        if verbose and model == "gpt":
            self.logger.info("### dataset example: " + data_with_prompt[0] + "\n")
            return data_with_prompt

        else:
            dataset = Dataset2.from_dict({"text": data_with_prompt})

            tokenized_datasets = dataset.map(
                lambda examples: tokenizer(
                    examples["text"],
                    # max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    #     add_special_tokens=True
                ),
                batched=True,
                num_proc=1,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

            if verbose:
                self.logger.info(
                    "### tokenized_datasets...example: " + tokenized_datasets["text"][0]
                )
            dataset = TextDataset(tokenized_datasets, tokenizer)

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=dataset.collater,
            )
            return iter(data_loader)

    def forward(self):
        raise NotImplementedError

    def get_generations(self, prompt_pre,  eval_src, ref_texts):
        args = self.args
        batch_size = args.batch_size
        dataset = self.create_dataset(
            args.dev_file,
            eval_src,
            ref_texts,
            tokenizer=self.tokenizer,
            model=self.model_name,
            batch_size=batch_size,
            prompt_num=args.prompt_num,
            prompt_pre=prompt_pre,
            task=args.task,
            position=args.position,
        )
        hypos = []
        if "gpt" in args.language_model:
            if args.task == "cls":
                hypos = llm_cls(
                    dataset=dataset,
                    client=self.client,
                    type="davinci",
                    batch_size=self.args.batch_size,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=0,
                )
            else:
                if "davinci" in args.llm_type:
                    hypos = llm_query(
                        dataset,
                        client=self.client,
                        type=args.llm_type,
                        task=True,
                        temperature=0,
                        **self.llm_config,
                    )
                    # hypos = llm_gen(dataset)
                else:
                    for data in tqdm(dataset):
                        pred = llm_query(
                            data,
                            client=self.client,
                            type=args.llm_type,
                            task=True,
                            **self.llm_config,
                        )
                        # print(pred)
                        hypos.append(pred)
        else:
            all_test_data = []
            try:
                while True:
                    cond = next(dataset)
                    all_test_data.append(cond)
            except StopIteration:
                # self.logger.info('### End of Loading datasets...')
                pass
            with torch.no_grad():
                for cond in tqdm(all_test_data):
                    input_ids_x = cond.pop("input_ids").to(self.device)
                    input_ids_mask = cond.pop("attention_mask").to(self.device)
                    prompt_len = cond.pop("prompt_len")

                    generate_ids = self.model.generate(
                        input_ids=input_ids_x,
                        max_new_tokens=args.max_new_tokens,
                        attention_mask=input_ids_mask,
                    )
                    generate_ids = generate_ids[:, prompt_len:-1]
                    pred = self.tokenizer.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                    )
                    # print(pred)
                    hypos.extend(pred)
        return hypos


class CLSEvaluator(Evaluator):
    def __init__(self, args):
        super(CLSEvaluator, self).__init__(args)
        self.verbalizers = get_dataset_verbalizers(args.dataset)
        if "gpt" not in args.language_model:
            self.verbalizer_ids = [
                self.tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers
            ]
        args.dev_file = (
            f"./data/cls/{args.dataset}/dev_{args.sample_num}.txt"
            if args.dev_file is None
            else args.dev_file
        )
        args.test_file = (
            f"/ml-dl/v-qingyanguo/data/cls/data/processed/{args.dataset}/test.txt"
            if args.test_file is None
            else args.test_file
        )

    def forward(
        self, prompt_pre="", eval_src=None, ref_texts=None, output=None
    ):
        args = self.args
        batch_size = args.batch_size
        hypos = []
        pred_mode = "logits" if "opt" in args.language_model else "gen"
        if "gpt" in args.language_model or pred_mode == "gen":
            ref_texts = (
                [ref[1:] for ref in ref_texts]
                if args.dataset not in ["agnews"]
                else ref_texts
            )

        if "gpt" in args.language_model:
            dataset = self.create_dataset(
                args.dev_file,
                eval_src,
                ref_texts,
                tokenizer=self.tokenizer,
                model=self.model_name,
                batch_size=batch_size,
                prompt_num=args.prompt_num,
                prompt_pre=prompt_pre,
                task="cls",
                position=args.position,
            )
            pred = llm_cls(
                dataset=dataset,
                client=self.client,
                type=args.llm_type,
                **self.llm_config,
            )
            hypos = list(
                map(
                    lambda x: first_appear_pred(
                        x, dataset_classes_list[args.dataset], self.logger
                    ),
                    pred,
                )
            )
            not_hit = 0
            for i in hypos:
                if i not in dataset_classes_list[args.dataset]:
                    not_hit += 1
            self.logger.info(f"not hit: {not_hit}, ratio: {not_hit/len(hypos)}")

        else:
            if pred_mode == "gen":
                pred = self.get_generations(
                    prompt_pre, eval_src, ref_texts
                )
                pred = [remove_punctuation(i) for i in pred]
                hypos = list(
                    map(
                        lambda x: first_appear_pred(
                            x, dataset_classes_list[self.args.dataset], self.logger
                        ),
                        pred,
                    )
                )

            elif pred_mode == "logits":
                all_test_data = []
                dataset = self.create_dataset(
                    args.dev_file,
                    eval_src,
                    ref_texts,
                    tokenizer=self.tokenizer,
                    model=self.model_name,
                    batch_size=batch_size,
                    prompt_num=args.prompt_num,
                    prompt_pre=prompt_pre,
                    task=args.task,
                    position=args.position,
                )
                try:
                    while True:
                        cond = next(dataset)
                        all_test_data.append(cond)
                except StopIteration:
                    pass
                with torch.no_grad():
                    for cond in tqdm(all_test_data):
                        pred = self.get_pred(cond)
                        pred = [self.verbalizers[i] for i in pred]
                        # print(pred)
                        hypos.extend(pred)

        score = cal_cls_score(hypos, ref_texts, metric="acc")
        return {"hypos": hypos, "scores": [score]}

    @torch.no_grad()
    def _get_logits(
        self,
        input_ids,
        input_ids_mask,
    ) -> torch.Tensor:
        logits = self.model(
            **{"input_ids": input_ids, "attention_mask": input_ids_mask}
        ).logits
        return logits

    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        return mask_token_index

    def get_pred(self, cond) -> np.ndarray:
        input_ids_x = cond.pop("input_ids").to(self.device)
        input_ids_mask = cond.pop("attention_mask").to(self.device)
        prompt_len = cond.pop("prompt_len")
        logits = self._get_logits(input_ids_x, input_ids_mask)  # (16. 71, 50265)
        if self.is_mask_lm:
            mask_token_indices = self._get_mask_token_index(input_ids_x)  # (16)
            # note here, for mask LM, we need to get the logits of the mask token
            # couldn't replace range(batch_size) with ":"
            out_logits = logits[
                range(logits.shape[0]), mask_token_indices, :
            ]  # (16, 16, 50265)
        else:
            out_logits = logits[range(logits.shape[0]), -1, :]  # (16, 50272)
        class_probs = torch.softmax(out_logits[:, self.verbalizer_ids], -1)
        # Get labels
        predicted_labels = torch.argmax(class_probs, dim=-1)
        return predicted_labels


class SumEvaluator(Evaluator):
    def __init__(self, args):
        super(SumEvaluator, self).__init__(args)
        self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sum_data(
            args.dataset, args.seed, args.sample_num
        )

    def forward(
        self, prompt_pre="",eval_src=None, ref_texts=None, output=None
    ):
        hypos = []
        hypos = self.get_generations(prompt_pre,eval_src, ref_texts)
        hypos = [hypo.replace("\n", "") for hypo in hypos]
        # print(len(hypos))
        for i in range(len(hypos)):
            if len(hypos[i]) == 0 or hypos[i].isspace():
                hypos[i] = eval_src[i]
                if len(eval_src[i]) == 0:
                    hypos[i] = "None"
        # print(hypos)
        if output:
            with open(output, "w") as f:
                for hypo in hypos:
                    f.write(hypo + "\n")
        rouge1, rouge2, rougel = cal_rouge(hypos, ref_texts)
        mean_score = np.mean([rouge1, rouge2, rougel])
        return {"hypos": hypos, "scores": [rouge1, rouge2, rougel, mean_score]}


class SimEvaluator(Evaluator):
    def __init__(self, args):
        super(SimEvaluator, self).__init__(args)
        
        args.dev_file = (
            f"./data/sim/{args.dataset}/dev_{args.sample_num}.txt"
            if args.dev_file is None
            else args.dev_file
        )
        self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sim_data(
            args.dataset, args.seed
        )

    def forward(
        self, prompt_pre="",  eval_src=None, ref_texts=None, output=None
    ):
        hypos = self.get_generations(prompt_pre, eval_src, ref_texts)
        sari_score = cal_sari(eval_src, hypos, ref_texts)
        return {"hypos": hypos, "scores": [sari_score]}
