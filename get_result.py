import argparse
import numpy as np
from utils import cal_mean_std

parser = argparse.ArgumentParser(description="src or tgt")
parser.add_argument('--path', '-p', required=True)
parser.add_argument('--multi_metric', '-m', action='store_true')
parser.add_argument('--step', '-s', default=9)

args = parser.parse_args()

def cal_test_result_3seed(path, step):
    scores = []
    for seed in [5, 10, 15]:
        with open(f'{path}/seed{seed}/step{step}_pop_test.txt') as f:
            line = f.readlines()[0].strip()
            score = line.split('\t')[-1]
            print(f'seed {seed}: {score}\n')
            score = float(score)
            if score < 1.0:
                score *= 100
            scores.append(score)
    print('mean, std:')
    print(cal_mean_std(scores))

def cal_dev_result_3seed(path,step):
    for seed in [5, 10, 15]:
        with open(f'{path}/seed{seed}/step{step}_pop.txt') as f:
            line = f.readlines()[-2:]
            best_score = line[0].strip().split('best score: ')[-1]
            avg_score = line[1].strip().split('average score: ')[-1]
            print(f'seed {seed}: best score: {best_score}, average score: {avg_score}\n')

def cal_sum_test_result_3seed(path,step):
    all_scores = []
    for seed in [5, 10, 15]:
        with open(f'{path}/seed{seed}/step{step}_pop_test.txt') as f:
            line = f.readlines()[0].strip()
            scores = line.split('\t')[2:]
            scores = [float(s) for s in scores]
            scores = [s * 100 if s < 1.0 else s for s in scores]
            print(f'seed {seed}: {scores}')
            all_scores.append(scores)
    all_scores = np.array(all_scores)
    avg = np.mean(all_scores, axis=0).tolist()
    std = np.std(all_scores, axis=0).tolist()
    avg_std = [(a, s) for a, s in zip(avg, std)]
    print('mean, std:')
    print(avg_std)

if __name__ == '__main__':
    step = args.step
    if args.multi_metric:
        cal_sum_test_result_3seed(args.path)
    else:
        print('dev result:\n')
        cal_dev_result_3seed(args.path, step)
        print('test result:')
        cal_test_result_3seed(args.path, step)