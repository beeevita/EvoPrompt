import random
import random
import numpy as np
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def batchify(data, batch_size=20):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data

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

def extract_ans(ans, mode):
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == 'free_form':
        if ans[-1] == '.':
            ans = ans[:-1]
        return ans

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_numbers(string):
    return [int(num) for num in re.findall(r'\d+', string)][0]