import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import math
import copy
import neptune
import pickle

from utils.datasets import get_dataloader, plot_images_stacked
from utils.networks import SimCLR, DecoderNN_1input, build_resnet18, build_resnet50
from utils.contrastive import InfoNCELoss, knn_evaluation, top_k_accuracy, eval_loop, get_avg_loss
from utils.ppo import (
    collect_trajectories_with_input,
    ppo_update_with_input,
    print_sorted_strings_with_counts
)
from utils.transforms import get_transforms_list, NUM_DISCREATE, transformations_dict, get_policy_distribution
from utils.logs import init_neptune, get_model_save_path
import argparse



def fix_seed(seed=None):
    
    if seed is None:
        seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def cuda_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a
    
    print(f'total memory: {t/1024**3:.4f}')
    print(f'reserved memory: {r/1024**3:.4f}')
    print(f'allocated memory: {a/1024**3:.4f}')
    print(f'free memory: {f/1024**3:.4f}')
    

def mean_last_percentage(lst, P):

    # Calculate the number of elements to consider based on the percentage
    num_elements = int(len(lst) * P)
    
    # Extract the last percentage of elements
    last_percentage_elements = lst[-num_elements:]

    # Calculate the mean
    mean_value = sum(last_percentage_elements) / len(last_percentage_elements)
    
    return mean_value


def ppo_init(args, device):

    decoder = DecoderNN_1input(
        transforms=list(transformations_dict.keys()),
        num_discrete_magnitude=NUM_DISCREATE,
        device=device
    )
    
    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=0.00005
    )

    return decoder, optimizer


def contrastive_init(args, device):
    
    if args.encoder_backbone == 'resnet18':
        encoder = build_resnet18()
    elif args.encoder_backbone == 'resnet50':
        encoder = build_resnet50()
    
    encoder = encoder.to(device)

    criterion = InfoNCELoss()
    
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        momentum=0.9,
        lr=args.lr * args.simclr_bs / 256,
        weight_decay=0.0005
    )
    
    return encoder, optimizer, criterion


def adjust_learning_rate(
        epochs: int,
        warmup_epochs: int,
        base_lr: float,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        step: int
    ):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def init(args, neptune_run, device):
    
    start_epoch = 1
    
    encoder, simclr_optimizer, simclr_criterion = contrastive_init(args, device)
    decoder, ppo_optimizer = ppo_init(args, device)
    
    if args.checkpoint_id:
        
        checkpoint_params = args.checkpoint_params
        
        encoder.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder.pt'))
        simclr_optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder_opt.pt'))
        decoder.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder.pt'))
        ppo_optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder_opt.pt'))
        
        prev_run = neptune.init_run(
            project="nazim-bendib/simclr-rl",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
            with_id=args.checkpoint_id
        )
        
        # [tag.split('=')[1] for tag in list(run['sys/tags'].fetch()) if 'model_save_path=' in tag][0]
        
        test_acc = prev_run['linear_eval/test_acc'].fetch_values().value.tolist()
        start_epoch = len(test_acc) + 1
        
        for acc in test_acc:
            neptune_run["linear_eval/test_acc"].append(acc)
            
        loss = prev_run['simclr/loss'].fetch_values().value.tolist()
        for i in loss:
            neptune_run['simclr/loss'].append(i)

        prev_run.stop()
    
    return (
        (encoder, simclr_optimizer, simclr_criterion),
        (decoder, ppo_optimizer), start_epoch
    )
    
        
def ppo_round(
        encoder: SimCLR,
        decoder: DecoderNN_1input,
        optimizer: torch.optim.Optimizer,
        args,
        avg_infoNCE_loss: tuple,
        neptune_run: neptune.Run
    ):
    
    ppo_rounds = args.ppo_iterations
    len_trajectory = args.ppo_len_trajectory 
    batch_size = args.ppo_collection_bs 
    ppo_epochs = args.ppo_update_epochs 
    ppo_batch_size = args.ppo_update_bs
    
    losses = []
    rewards = []
    
    tqdm_range = tqdm(range(ppo_rounds), desc='[ppo_round]')
    for round_ in tqdm_range:
    
        trajectory, (img1, img2, new_img1, new_img2), entropy = collect_trajectories_with_input(
            len_trajectory=len_trajectory,
            encoder=encoder,
            decoder=decoder,
            avg_infoNCE_loss=avg_infoNCE_loss,
            args=args,
            batch_size=batch_size,
            neptune_run=neptune_run
        )

        loss = ppo_update_with_input(
            trajectory,
            decoder,
            optimizer,
            ppo_epochs=ppo_epochs,
            ppo_batch_size=ppo_batch_size,
        )
        
        losses.append(loss)
        rewards.append(float(trajectory[-1].mean()))
                
    stored_actions_index = trajectory[1]
    transforms_list_1, transforms_list_2 = get_transforms_list(
        stored_actions_index,
        num_magnitudes=decoder.num_discrete_magnitude
    )
    string_transforms = []
    for trans1, trans2 in zip(transforms_list_1, transforms_list_2):
        s1 = ' '.join([ f'{name[:4]}_{round(magnetude, 3)}' for (name, _, magnetude) in trans1])
        s2 = ' '.join([ f'{name[:4]}_{round(magnetude, 3)}' for (name, _, magnetude) in trans2])
        string_transforms.append( f'{s1}  ||  {s2}' )
    print_sorted_strings_with_counts(string_transforms, topk=5)
        
    
    return trajectory, (img1, img2, new_img1, new_img2), entropy, (losses, rewards)


def contrastive_round(
        encoder: SimCLR,
        policies: list,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        random_p: float,
        args,
        epoch: int,
        neptune_run: neptune.Run,
        device
    ):
    
    batch_size = args.simclr_bs
    
    dist = get_policy_distribution(N=min(len(policies), 4), p=0.6)
    print(f'policies dist: {dist}')
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=batch_size,
        random_p=random_p,
        policies=policies,
        ppo_dist=dist
    )
    
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader), desc='[contrastive_round]')    
    
    encoder.train()
    lr = None
    for it, (x, x1, x2, y) in tqdm_train_loader:
        
        # plot_images_stacked(x1, x2)
        
        lr = adjust_learning_rate(epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr * args.simclr_bs / 256,
            optimizer=optimizer,
            loader=train_loader,
            step=it+(epoch-1)*len(train_loader)
        )
        
        # Simclr:
        _, z1 = encoder(x1.to(device))
        _, z2 = encoder(x2.to(device))

        sim, _, simclr_loss = criterion(z1, z2, temperature=0.5)
        
        optimizer.zero_grad()
        simclr_loss.backward()
        optimizer.step()

        # logs:
        neptune_run["simclr/loss"].append(simclr_loss.item())
        neptune_run["simclr/top_5_acc"].append(top_k_accuracy(sim, 5))
        neptune_run["simclr/top_1_acc"].append(top_k_accuracy(sim, 1))

        
        

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    
    neptune_run = init_neptune(
        tags=[
            f'random_p={args.random_p}', 
            f'model_save_path={args.model_save_path}', 
            f'reward_a={args.reward_a}', 
            f'reward_b={args.reward_b}', 
            f'encoder_backbone={args.encoder_backbone}', 
            f'lr={args.lr}',
        ],
        mode=args.mode,
    )
    neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])


    (encoder, simclr_optimizer, simclr_criterion), (decoder, ppo_optimizer), start_epoch = init(args, neptune_run, device)

    all_policies = []


    for epoch in tqdm(range(start_epoch, args.epochs+1), desc='[Main Loop]'):
        
        random_p = 1 if epoch <= args.warmup_epochs else args.random_p
        # random_p = args.random_p
        print(f'EPOCH:{epoch}    P:{random_p}')
            
        
        if (epoch > args.warmup_epochs) and ((epoch-1) % 10 == 0):
            
            avg_infoNCE_loss = get_avg_loss(
                encoder=encoder,
                policies=all_policies,
                criterion=simclr_criterion,
                random_p=random_p if (epoch-1) > args.warmup_epochs else 1,
                batch_size=args.ppo_collection_bs,
                args=args,
                num_steps=2
            )
            print(f"avg_infoNCE: {avg_infoNCE_loss}")
                    
            decoder, ppo_optimizer = ppo_init(args, device)
            trajectory, (img1, img2, new_img1, new_img2), entropy, (ppo_losses, ppo_rewards) = ppo_round(
                encoder=encoder, 
                decoder=decoder,
                optimizer=ppo_optimizer,
                args=args,
                avg_infoNCE_loss=avg_infoNCE_loss,
                neptune_run=neptune_run,
            )
                    
            policy = decoder.get_policy_list()
            all_policies.append(policy)
                
            with open(f'{model_save_path}/all_policies.pkl', 'bw') as file:
                pickle.dump(all_policies, file)
        
        
        
        
        contrastive_round(
            encoder=encoder,
            policies=all_policies,
            epoch=epoch,
            args=args,
            optimizer=simclr_optimizer, 
            criterion=simclr_criterion, 
            random_p=random_p,
            neptune_run=neptune_run,
            device=device
        )
        

        if epoch % 1 == 0:
            test_acc = knn_evaluation(encoder, args)
        neptune_run["linear_eval/test_acc"].append(test_acc)

        
        
        if epoch in [200, 400, 600, 800]:
            os.mkdir(f'{model_save_path}/epoch_{epoch}/')
            torch.save(encoder.state_dict(), f'{model_save_path}/epoch_{epoch}/encoder.pt')
            torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/epoch_{epoch}/encoder_opt.pt')
            torch.save(decoder.state_dict(), f'{model_save_path}/epoch_{epoch}/decoder.pt')
            torch.save(ppo_optimizer.state_dict(), f'{model_save_path}/epoch_{epoch}/decoder_opt.pt')
            
            neptune_run[f"params/epoch_{epoch}/encoder"].upload(f'{model_save_path}/epoch_{epoch}/encoder.pt')
            neptune_run[f"params/epoch_{epoch}/encoder_opt"].upload(f'{model_save_path}/epoch_{epoch}/encoder_opt.pt')
            neptune_run[f"params/epoch_{epoch}/decoder"].upload(f'{model_save_path}/epoch_{epoch}/decoder.pt')
            neptune_run[f"params/epoch_{epoch}/decoder_opt"].upload(f'{model_save_path}/epoch_{epoch}/decoder_opt.pt')
        
        
        if (epoch % 10 == 0) or (epoch == args.epochs):
            torch.save(encoder.state_dict(), f'{model_save_path}/encoder.pt')
            torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/encoder_opt.pt')
            torch.save(decoder.state_dict(), f'{model_save_path}/decoder.pt')
            torch.save(ppo_optimizer.state_dict(), f'{model_save_path}/decoder_opt.pt')
            
            neptune_run["params/encoder"].upload(f'{model_save_path}/encoder.pt')
            neptune_run["params/encoder_opt"].upload(f'{model_save_path}/encoder_opt.pt')
            neptune_run["params/decoder"].upload(f'{model_save_path}/decoder.pt')
            neptune_run["params/decoder_opt"].upload(f'{model_save_path}/decoder_opt.pt')
            if all_policies:
                neptune_run["params/policies"].upload(f'{model_save_path}/all_policies.pkl')



    print('Linear evaluation man')
    accs = []
    for i in range(5):
        accs.append(eval_loop(copy.deepcopy(encoder.enc), args, i))
        line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
        print(line_to_print)

        

    neptune_run.stop()
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument Parser for Training Configuration')
        
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs')

    parser.add_argument('--simclr_iterations', type=str, default='all', help='Iterations for SimCLR training')
    parser.add_argument('--simclr_bs', type=int, default=512, help='Batch size for SimCLR training')
    parser.add_argument('--linear_eval_epochs', type=int, default=200, help='Number of epochs for linear evaluation')
    parser.add_argument('--random_p', type=float, default=1.0, help='Random probability')
    parser.add_argument('--encoder_backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help='Encoder backbone architecture')
    parser.add_argument('--dataset', type=str, default='svhn', choices=['cifar10', 'svhn', 'TinyImagenet'], help='Dataset')

    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
        
    parser.add_argument('--ppo_iterations', type=int, default=100, help='Number of iterations for PPO')
    parser.add_argument('--ppo_len_trajectory', type=int, default=128, help='Length of trajectory for PPO')
    parser.add_argument('--ppo_collection_bs', type=int, default=128, help='Batch size for PPO data collection')
    parser.add_argument('--ppo_update_bs', type=int, default=16, help='Batch size for PPO update')
    parser.add_argument('--ppo_update_epochs', type=int, default=4, help='Number of epochs for PPO update')
    parser.add_argument('--reward_a', type=float, default=1.4, help='Reward parameter a for PPO')
    parser.add_argument('--reward_b', type=float, default=0.2, help='Reward parameter b for PPO')

    parser.add_argument('--mode', type=str, default='debug', choices=['async', 'debug'], help='Training mode')

    parser.add_argument('--model_save_path', type=str, default="", help='Path to save the model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')

    parser.add_argument('--checkpoint_id', type=str, default="", help='Checkpoint ID')
    parser.add_argument('--checkpoint_params', type=str, default="", help='Checkpoint parameters')
    # Uncomment below lines if you want to set default values for checkpoint_id and checkpoint_params
    # parser.add_argument('--checkpoint_id', type=str, default="SIM-511", help='Checkpoint ID')
    # parser.add_argument('--checkpoint_params', type=str, default="params_655", help='Checkpoint parameters')

    args = parser.parse_args()
    
    seed = fix_seed(args.seed)
    
    if not args.model_save_path:
        model_save_path = get_model_save_path()
        args.model_save_path = model_save_path
        
    
    args.dataset = 'cifar10' # ['cifar10', 'svhn', 'TinyImagenet']
        
    args.epochs = 800
    args.warmup_epochs = 10

    args.simclr_bs = 4
    args.random_p = 1.0
    args.lr = 0.03
        
    args.ppo_iterations = 2
    args.ppo_len_trajectory = 16
    args.ppo_collection_bs = 4
    args.ppo_update_bs = 16
    args.ppo_update_epochs = 4
    args.reward_a = 1.4
    args.reward_b = 0.2

    args.mode = 'debug' # ['async', 'debug']

    args.model_save_path = model_save_path
    args.seed = seed

    args.checkpoint_id = ""
    args.checkpoint_params = ""


    main(args)