import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='training args.')

    # prompt args
    parser.add_argument('--task', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='batchsize in decoding. Left padding in default')
    parser.add_argument('--max-new-tokens',
                        type=int,
                        default=128,
                        help='max new tokens to generate by the model')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='output file name')

    parser.add_argument(
        '--sample_num',
        type=int,
        default=100,
        help='number of samples used to choose the optimized sequences')

    # DE args
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--popsize', type=int, default=10)
    parser.add_argument('--evo_mode', type=str, default='de', help='mode of the evolution', choices=['de', 'ape', 'ga',])
    parser.add_argument('--donor_random',action='store_true')
    parser.add_argument(
        "--sel_mode", type=str, choices=["wheel", "random", "tour"], default="wheel", help='selection strategy for parents, only used for GA'
    )

    parser.add_argument('--llm_type', type=str, default='davinci')
    parser.add_argument('--initial', type=str, default='cot', choices=['cot', 'desc', 'all', 'ckpt'], help='the stylf of the prompt, cot (task agnostic): Let\'s think step by step., desc (task specific): the description of the task', )
    parser.add_argument('--initial_mode', type=str)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--para_mode', type=str, default=None)
    parser.add_argument('--template', type=str, default='v1')
    parser.add_argument('--client', action='store_true'),

    parser.add_argument('--cot_cache_path', type=str, default=None)
    parser.add_argument('--desc_cache_path', type=str, default=None)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--setting', type=str, default='default')
    parser.add_argument("--ga_mode", type=str, default="topk", help="update strategy for GA")
    parser.add_argument('--content', type=str, default='')
    parser.add_argument('--ckpt_pop', type=str, default=None)
    parser.add_argument('--demon', type=int, default=1, help='few-shot or zero-shot', choices=[0,1])
    parser.add_argument(
        "--content",
        type=str,
        default="",
        help="content of the prompt, used when testing single prompt",
    )
    args = parser.parse_args()
    return args