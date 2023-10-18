import json
import sys
from args import parse_args
from evaluator import CLSEvaluator,  SumEvaluator, SimEvaluator
from utils import load_cls_data

sys.path.append("./")


def evaluate_single_prompt(evaluator, args, prompt_pre, output=None):
    if args.task == 'cls':
        eval_src, eval_tgt = load_cls_data(
            evaluator.verbalizers, args.test_file
        )
    else:
        eval_src, eval_tgt = evaluator.test_src, evaluator.test_tgt
    res = evaluator.forward(prompt_pre, eval_src, eval_tgt, output=output)
    return res["scores"]


def eval_cls_baseline(evaluator,mode, args):
    dataset =  args.dataset
    instructions = json.load(open("./data/cls/baseline.json", "r"))
    prompt = instructions[mode][dataset]
    print(f"evaluating instruction {prompt}")
    scores = evaluate_single_prompt(evaluator, args, prompt)
    print(scores)
    return scores

def evaluate_optimized_prompt(population, pop_marks, out_path, evaluator, args):
    with open(
        out_path,
        "w",
    ) as wf:
        prompts, marks, all_scores, scores_strs = [], [], [], []
        if args.task == 'cls':
            eval_src, eval_tgt = load_cls_data(
                evaluator.verbalizers, args.test_file
            )
        else:
            eval_src, eval_tgt = evaluator.test_src, evaluator.test_tgt

        for prompt, mark in zip(population, pop_marks):
            res = evaluator.forward(prompt,eval_src, eval_tgt)
            scores = res["scores"]
            all_scores.append(scores[-1])
            scores_str = "\t".join([str(round(s, 4)) for s in scores])
            wf.write(f"{mark}\t{prompt}\t{scores_str}\n")
            scores_strs.append(scores_str)
            marks.append(mark)
            prompts.append(prompt)
            wf.flush()
        score_sorted, prompts_sorted, mark_sorted, scores_strs_sorted = (
            list(t)
            for t in zip(
                *sorted(zip(all_scores, prompts, marks, scores_strs), reverse=True)
            )
        )

        wf.write("\n----------sorted results----------\n")
        for i in range(len(score_sorted)):
            wf.write(
                f"{mark_sorted[i]}\t{prompts_sorted[i]}\t{scores_strs_sorted[i]}\n"
            )
        wf.close()
    return score_sorted[0], prompts_sorted[0]
    
if __name__ == "__main__":
    eval_args = parse_args()
    print(eval_args)

    task2evaluator = {
        "cls": CLSEvaluator,
        "sum": SumEvaluator,
        "sim": SimEvaluator,
    }
    # ICL
    scores = []
    evaluator = task2evaluator[eval_args.task](eval_args)
    scores = evaluate_single_prompt(
       evaluator, eval_args, eval_args.content,
    )
    print(scores)

