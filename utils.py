import re
import yaml
import random
import string
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import torch
from typing import List
import numpy as np

dataset_classes_list = {
    'sst2': ['positive', 'negative'],
    'mr': ['positive', 'negative'],
    'cr': ['positive', 'negative'],
    'subj': ['subjective', 'objective'],
    'agnews': ['World', 'Sports', 'Business', 'Tech'],
    'trec': ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number'],
    'sst-5': ['terrible', 'bad', 'okay', 'good', 'great'],
}


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)

def first_appear_pred(text, verbalizer_dict, logger):
    text = text.lower()
    verbalizer_dict = [k.lower() for k in verbalizer_dict]
    for word in text.split():
        if word in verbalizer_dict:
            return word
    # logger.info("cannot decode {}".format(text))
    return ""


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]


def json2list(file):
    with open(file, 'r') as f:
        lines = json.load(f)
    return lines


def format_template(
    src,
    tgt="",
    template="",
    src_name="",
    tgt_name="",
    line_break='\n',
):

    template_ = template
    if isinstance(tgt, list):
        tgt = tgt[0]
    template_ = template_.replace("<input>", src).replace("<output>", tgt)
    template_ = template_.replace("<line_break>", line_break)
    return template_

def get_final_prompt(text):
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith("\"") and text.endswith("\""):
            text = text[1:-1]
        return text


def load_cls_data(verbalizers=None, data_path=None,  sample_indices=None):
    test_data = read_lines(
        data_path, sample_indices=sample_indices)
    test_src = []
    test_tgt = []
    for i, line in enumerate(test_data):
        try:
            cur_src, cur_tgt = line.split('\t')
        except:
            raise ValueError
        test_src.append(cur_src)
        test_tgt.append(verbalizers[int(cur_tgt)])
    return test_src, test_tgt


def load_sum_data_(src_file, tgt_file, sample_indices=None):
    src = read_lines(src_file, sample_indices=sample_indices)
    tgt = read_lines(tgt_file, sample_indices=sample_indices)
    return src, tgt

def load_sum_data(dataset, seed, sample_num):
    random.seed(seed)
    if dataset == 'sam':
        dev_file = './data/sum/sam/valid'
        test_file = './data/sum/sam/test'
        dev_src, dev_tgt = load_sum_data_(f'{dev_file}.src',f'{dev_file}.tgt')
        test_src, test_tgt = load_sum_data_(f'{test_file}.src',f'{test_file}.tgt')
        sample_indices = random.sample(range(len(dev_src)), sample_num)
    dev_src = [dev_src[i] for i in sample_indices]
    print(sample_indices)
    dev_tgt = [dev_tgt[i] for i in sample_indices]
    return dev_src, dev_tgt, test_src, test_tgt

def load_sim_data_(src_file, tgt_files, sample_indices=None):
    src = read_lines(src_file, sample_indices=sample_indices)
    tgt = []
    for tgt_file in tgt_files:
        tgt.append(read_lines(tgt_file, sample_indices=sample_indices))
    print(len(src))
    print(len(tgt))
    return src, tgt


def load_sim_data(dataset, seed):
    random.seed(seed)
    if dataset == 'asset':
        dev_src_file = './data/sim/asset/dev/asset.valid.src'
        dev_tgt_files = [
            f'./data/sim/asset/dev/asset.valid.simp.{i}' for i in range(10)]
        test_src_file = './data/sim/asset/test/asset.test.src'
        test_tgt_files = [
            f'./data/sim/asset/test/asset.test.simp.{i}' for i in range(10)]
    else:
        raise ValueError("dataset not supported")

    dev_src, dev_tgt = load_sim_data_(dev_src_file, dev_tgt_files)
    test_src, test_tgt = load_sim_data_(test_src_file, test_tgt_files)
    sample_indices = random.sample(range(len(dev_src)), 100)
    print(sample_indices)
    dev_src = [dev_src[i] for i in sample_indices]
    dev_tgt_ = []
    for i in dev_tgt:
        dev_tgt_.append([i[j] for j in sample_indices])
    return dev_src, dev_tgt_, test_src, test_tgt

def extract_numbers(string):
    return [int(num) for num in re.findall(r'\d+', string)][0]

def extract_n_samples_per_class(src, tgt, n, dataset):
    src_new = []
    tgt_new = []
    for label in set(tgt):
        cur_src = [src[i] for i, value in enumerate(tgt) if value == label]
        cur_tgt = [tgt[i] for i, value in enumerate(tgt) if value == label]
        rand_indices = random.sample(range(len(cur_src)), n)
        # print(rand_indices)
        src_new += [cur_src[i] for i in rand_indices]
        tgt_new += [cur_tgt[i] for i in rand_indices]
    tgt_new = [e[1:] for e in tgt_new] if dataset != 'agnews' else tgt_new
    return src_new, tgt_new

def batchify(data, batch_size=16):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_log(log_path, log_name="basic"):
    print("Setting up log for", log_name)
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        # log_path = os.path.join("logs", log_name)
        logger.setLevel(logging.DEBUG)
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        stream_handler = logging.StreamHandler()
        # formatter = logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s")
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger



def get_dataset_verbalizers(dataset: str) -> List[str]:
    if dataset in ["sst2", "yelp-2", "mr", "cr"]:
        verbalizers = ["\u0120negative", "\u0120positive"]  # num_classes
        # verbalizers = ['\u0120terrible', '\u0120great']  # num_classes
    elif dataset == "agnews":
        verbalizers = ["World", "Sports", "Business", "Tech"]  # num_classes
    elif dataset in ["sst-5", "yelp-5"]:
        verbalizers = [
            "\u0120terrible",
            "\u0120bad",
            "\u0120okay",
            "\u0120good",
            "\u0120great",
        ]  # num_classes
    elif dataset == "subj":
        verbalizers = ["\u0120subjective", "\u0120objective"]
    elif dataset == "trec":
        verbalizers = [
            "\u0120Description",
            "\u0120Entity",
            "\u0120Expression",
            "\u0120Human",
            "\u0120Location",
            "\u0120Number",
        ]
    return verbalizers


def k_init_pop(initial_mode, init_population, k):
    if initial_mode == "topk":
        population = [i for i in init_population[:k]]
    elif initial_mode == "para_topk":
        population = [i for i in init_population[: k // 2]]
    elif initial_mode == "para_bottomk":
        population = [i for i in init_population[-k // 2 :]]
    elif initial_mode == "para_randomk":
        population = random.sample(init_population, k // 2)
    elif initial_mode == "randomk":
        population = random.sample(init_population, k)
    elif initial_mode == "bottomk":
        population = [i for i in init_population[-k:]]
    return population

def cal_mean_std(results):
    if results[0] < 1.0:
        results = [result * 100 for result in results]
    mean = np.mean(results)
    std = np.std(results)
    return round(mean, 2), round(std, 2)


if __name__ == '__main__':
    dev_src, dev_tgt, test_src, test_tgt = load_sum_data('sam', 5, 100)
    lengths = [len(i) for i in dev_src]
    from collections import Counter
    print(dict(Counter(lengths))[0])