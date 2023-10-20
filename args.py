import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="training args.")
    # prompt args
    parser.add_argument("--dataset", type=str, default="sst2", help="dataset name")
    parser.add_argument("--task", type=str, choices=["cls", "sum", "sim"])
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batchsize in decoding. Left padding in default",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="max new tokens to generate by the model",
    )
    parser.add_argument("--prompt-num", type=int, default=0, help="number of demonstrations,used for the in-context learning setting")
    parser.add_argument("--dev_file", type=str, default=None, help="dev set path")
    parser.add_argument("--output", type=str, default=None, help="output path")
    parser.add_argument(
        "--language_model",
        type=str,
        help="model for task implementation, e.g., alpaca, gpt",
    )
    parser.add_argument("--position", type=str, default="pre")
    parser.add_argument(
        "--sample_num",
        type=int,
        default=100,
        help="number of samples used to choose the optimized sequences",
    )
    parser.add_argument("--seed", type=int, default=5, help="random seed")
    # DE args
    parser.add_argument(
        "--budget", type=int, default=10, help="number of steps for evolution"
    )
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument(
        "--evo_mode",
        type=str,
        default="de",
        help="mode of the evolution",
        choices=["de", "ga"],
    )
    parser.add_argument("--llm_type", type=str, default="davinci", help='llm to generate prompt', choices=['davinci', 'turbo', 'gpt4'])
    parser.add_argument(
        "--initial",
        type=str,
        default="all",
        choices=["ape", "all", "ckpt"],
    )
    parser.add_argument("--initial_mode", type=str)
    parser.add_argument("--para_mode", type=str, default=None)
    parser.add_argument("--ckpt_pop", type=str, default=None)
    parser.add_argument("--template", type=str, default="v1", help='the template used for DE')
    parser.add_argument("--pred_mode", type=str, default="logits")
    parser.add_argument("--client", action="store_true"),
    parser.add_argument("--cache_path", type=str, default=None, help="cache path of the prompt score")
    parser.add_argument("--setting", type=str, default="default", help="setting of the OpenAI API")
    parser.add_argument("--donor_random", action="store_true", help='prompt 3 random or best, used only for DE')
    parser.add_argument("--ga_mode", type=str, default="topk", help="update strategy for GA")
    parser.add_argument(
        "--content",
        type=str,
        default="",
        help="content of the prompt, used when testing single prompt",
    )
    parser.add_argument("--write_step", type=int, default=10)
    parser.add_argument(
        "--sel_mode", type=str, choices=["wheel", "random", "tour"], default="wheel", help='selection strategy for parents, only used for GA'
    )
    parser.add_argument
    args = parser.parse_args()
    return args
