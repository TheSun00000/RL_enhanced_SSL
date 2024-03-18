import torch
import sys
import numpy as np
import copy
from utils.contrastive import eval_loop
from utils.networks import build_resnet18, build_resnet50
from utils.logs import init_neptune
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argument Parser for Training Configuration')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn', 'TinyImagenet'], help='Dataset')
    parser.add_argument('--param', type=str, help='Dataset')
    parser.add_argument('--neptune', type=bool, default=False, help='Dataset')
    args = parser.parse_args()

    ids = list(map(int, args.param.split(" ")))
    
    print(f'dataset: {args.dataset}')
    print(f'params: {ids}')
    
    if args.neptune:
        neptune_run = init_neptune(
            tags=['linear_eval'],
            mode='debug'
        )
        neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])

    for id in ids:

        
        path = f'params/params_{id}/encoder.pt'
        print('Checkpoint:', path)
        
        encoder = build_resnet50()
        # encoder = build_resnet18()
        encoder.load_state_dict(torch.load(path))
        encoder = encoder.to(device)

        accs = []
        for i in range(2):
            accs.append(eval_loop(copy.deepcopy(encoder.enc), args, i))
            line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
            print(line_to_print)
        
        
        print('\n\n\n\n\n\n\n\n')

    if args.neptune:
        neptune_run.stop()