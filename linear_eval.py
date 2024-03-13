import torch
import sys
import numpy as np
import copy
from utils.contrastive import eval_loop
from utils.networks import build_resnet18, build_resnet50
from utils.logs import init_neptune

device = 'cuda' if torch.cuda.is_available() else 'cpu'


    
    

neptune_run = init_neptune(
    tags=['linear_eval'],
    mode='debug'
)
neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])


ids = [673]

for id in ids:

    
    path = f'params/params_{id}/encoder.pt'
    print('Checkpoint:', path)
    
    encoder = build_resnet50()
    # encoder = build_resnet18()
    encoder.load_state_dict(torch.load(path))
    encoder = encoder.to(device)

    accs = []
    for i in range(2):
        accs.append(eval_loop(copy.deepcopy(encoder.enc), i))
        line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
        print(line_to_print)
    
    
    print('\n\n\n\n\n\n\n\n')

neptune_run.stop()