import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter
import neptune
from copy import deepcopy
import random


from utils.datasets import get_dataloader
from utils.transforms import (
    get_transforms_list,
    apply_transformations,
    NUM_DISCREATE,
    get_policy_distribution
)
from utils.contrastive import InfoNCELoss
from utils.networks import SimCLR, DecoderNN_1input


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

infonce_reward = InfoNCELoss(reduction='none')


def standarize_reward(reward, min, max):
    return (reward - min)/(max-min)

def similariy_reward_function(new_z1, new_z2):
    return - (F.normalize(new_z1) * F.normalize(new_z2)).sum(axis=-1)

def infonce_reward_function(new_z1, new_z2):
    bs = new_z1.shape[0]
    full_similarity_matrix, logits, loss = infonce_reward(new_z1, new_z2, temperature=0.5)
    reward = (loss[:bs] + loss[bs:]) / 2
    return reward


def sub_policy_stregnth(sub_policy):
    
    transform_weight = {
        'ShearX':0.5,
        'ShearY':0.5,
        'TranslateX':0.5,
        'TranslateY':0.5,
        'Rotate':0.5,
        'AutoContrast':0.9,
        'Invert':0.9,
        'Equalize':0.9,
        'Solarize':1.,
        'Posterize':1.,
        'Contrast':1.,
        'Color':1.,
        'Brightness':1.,
        'Sharpness':1.,
        'Cutout':0.5,
        'Identity':0.0
    }

    w = 0
    for name, pr, lvl in sub_policy:
        if name in ['ShearX', 'ShearY', 'Rotate', 'Contrast', 'Color', 'Brightness', 'Sharpness']:
            s = abs(lvl - 0.5) / 0.5
        # elif name in ['AutoContrast', 'Invert', 'Equalize']:
        #     s = 1
        elif name in ['Solarize', 'Posterize']:
            s = 1-lvl
        else:
            s = 0
        
        if s != 0:
            w += s * transform_weight[name]

    return w / len(sub_policy)


def get_action_strength(actions_index):
    
    ret = []
    
    for subpolicy_2 in actions_index:
        w = 0
        for subpolicy in subpolicy_2:
            w += sub_policy_stregnth(subpolicy)
        w /= len(subpolicy_2)
        ret.append(w)
    
    return torch.tensor(ret)



def print_sorted_strings_with_counts(input_list, topk):
    # Count occurrences of each string
    string_counts = Counter(input_list)

    # Sort strings by counts in descending order
    sorted_strings = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)[:topk]

    # Print the sorted strings with counts
    for i, (string, count) in enumerate(sorted_strings):
        print(f"{i}) {string}: {count}")


# #########################################################################################################################


def collect_trajectories_with_input(
        len_trajectory: int,
        encoder: SimCLR,
        decoder: DecoderNN_1input,
        args,
        avg_infoNCE_loss: float,
        batch_size: int,
        neptune_run: neptune.Run,
    ):

    assert len_trajectory % batch_size == 0
    
    data_loader = get_dataloader(
        args=args,
        batch_size=batch_size,
        transform=False,
    )
    
    last_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])


    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_actions_index = []
    stored_rewards = torch.zeros((len_trajectory,))

    
    mean_rewards = 0
    mean_strength = 0
    mean_entropy = 0
    mean_infonce_reward = 0 
    
    encoder.eval()
    
    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):

        begin, end = i*batch_size, (i+1)*batch_size
        
        try:
            img, y = next(data_loader_iterator)
        except StopIteration:
            iterator = iter(data_loader)
            img, y = next(iterator)

        with torch.no_grad():
            log_p, actions_index, entropy = decoder(batch_size=batch_size)
            # new_log_p, new_actions_index, new_entropy = decoder(batch_size=batch_size, old_action=actions_index)
            # assert (log_p == new_log_p).all()
            # assert (actions_index == new_actions_index)
            # assert (entropy == new_entropy).all()
            
        if args.ppo_permutation_p == 0:
            
            transforms_list_1, transforms_list_2 = get_transforms_list(actions_index)
            
            new_img1 = apply_transformations(img, transforms_list_1)
            new_img2 = apply_transformations(img, transforms_list_2)

            new_img1 = torch.stack([last_transform(img) for img in new_img1])
            new_img2 = torch.stack([last_transform(img) for img in new_img2])
                    
            with torch.no_grad():
                _, new_z1 = encoder(new_img1.to(device))
                _, new_z2 = encoder(new_img2.to(device))
                
            infoNCE_reward = infonce_reward_function(new_z1, new_z2)
        
        else: # random permute p% transformation:
            
            ids = torch.tensor([ range(batch_size), range(batch_size) ])
            new_ids = ids.clone()
            new_action_index = deepcopy(actions_index)
                
            L = int(batch_size * args.ppo_permutation_p)
            
            # Get random indices of the transformations that are going to be permuted
            indices = list(range(batch_size))
            random.shuffle(indices)
            indices = sorted(indices[:L])
            
            # print(indices)
            
            # original transformations indices of format: (branch[0,1], n) 
            indices_2d = [(0, i) for i in indices] + [(1, i) for i in indices]
            
            # shuffle  the original transformations indices
            new_indices_2d = indices_2d.copy()
            random.shuffle(new_indices_2d)
            
            # Permute the samples
            for (i, j), (new_i, new_j) in zip(indices_2d, new_indices_2d):
                # print((i, j), (new_i, new_j))
                new_ids[i, j] = ids[new_i, new_j]
                new_action_index[j][i] = new_action_index[new_j][new_i]


            transforms_list_1, transforms_list_2 = get_transforms_list(new_action_index)
            
            new_img1 = apply_transformations(img, transforms_list_1)
            new_img2 = apply_transformations(img, transforms_list_2)

            new_img1 = torch.stack([last_transform(img) for img in new_img1])
            new_img2 = torch.stack([last_transform(img) for img in new_img2])
                    
            with torch.no_grad():
                _, new_z1 = encoder(new_img1.to(device))
                _, new_z2 = encoder(new_img2.to(device))
                
            infoNCE_reward = infonce_reward_function(new_z1, new_z2)
            infoNCE_reward = (infoNCE_reward[new_ids[0]] + infoNCE_reward[new_ids[1]]) / 2
        
        
        
        strength = get_action_strength(actions_index)
        
        a, b = args.reward_a, args.reward_b
        infoNCE_reward_avg = infoNCE_reward/avg_infoNCE_loss
        reward = torch.where(infoNCE_reward_avg <= a, infoNCE_reward_avg, (-a/b)*(infoNCE_reward_avg-(a+b)))
        # reward = torch.where(infoNCE_reward_avg <= a, infoNCE_reward_avg+0.01*strength.to(device), (-a/b)*(infoNCE_reward_avg-(a+b)))
        # reward = torch.clip(infoNCE_reward_avg, max=a)
        
        
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_actions_index += actions_index
        stored_rewards[begin:end] = reward
        
        mean_rewards += reward.mean().item()
        mean_entropy += entropy.item()
        
        if infoNCE_reward is not None:
            mean_infonce_reward += infoNCE_reward.mean().item()

    
    # string_transforms = []
    # for trans1, trans2 in zip(transforms_list_1, transforms_list_2):
    #     s1 = ' '.join([ f'{name[:4]}_{round(level, 3)}' for (name, _, level) in trans1])
    #     s2 = ' '.join([ f'{name[:4]}_{round(level, 3)}' for (name, _, level) in trans2])
    #     string_transforms.append( f'{s1}  ||  {s2}' )
    # print_sorted_strings_with_counts(string_transforms, topk=5)
        
    mean_rewards /= (len_trajectory // batch_size)
    mean_infonce_reward /= (len_trajectory // batch_size)
    mean_strength /= (len_trajectory // batch_size)
    mean_entropy /= (len_trajectory // batch_size)
    
    neptune_run["ppo/reward"].append(mean_rewards)
    # neptune_run["ppo/mean_entropy"].append(mean_entropy)
    # if infoNCE_reward is not None:
        # neptune_run["ppo/infonce_reward"].append(mean_infonce_reward)

    # print('mean_entropy:', mean_entropy)
    
    return (
            (stored_log_p),
            (stored_actions_index),
            stored_rewards
        ), (img, img, new_img1, new_img2), entropy


def shuffle_trajectory(trajectory):

    (
        stored_log_p,
        actions,
        stored_rewards
    ) = trajectory

    permutation = torch.randperm(stored_log_p.shape[0])

    permuted_stored_log_p = stored_log_p[permutation]
    permuted_stored_actions  = [actions[i] for i in permutation]
    permuted_stored_rewards = stored_rewards[permutation]

    permuted_trajectory = (
        permuted_stored_log_p,
        permuted_stored_actions,
        permuted_stored_rewards
    )

    return permuted_trajectory


def ppo_update_with_input(
        trajectory: tuple,
        decoder: DecoderNN_1input,
        optimizer: torch.optim.Optimizer,
        ppo_batch_size:int=256,
        ppo_epochs:int=4
    ):


    for _ in range(ppo_epochs):
    
        shuffled_trajectory = shuffle_trajectory(trajectory)

        (
            (stored_log_p),
            (stored_actions_index),
            stored_rewards
        ) = shuffled_trajectory
        
        len_trajectory = stored_log_p.shape[0]


        assert len_trajectory % ppo_batch_size == 0

        cum_loss = 0
        cum_loss_counter = 0
        for i in range(len_trajectory // ppo_batch_size):

            begin, end = i*ppo_batch_size, (i+1)*ppo_batch_size

            old_log_p = stored_log_p[begin:end].to(device).detach()
            actions_index = stored_actions_index[begin:end]
            reward = stored_rewards[begin:end].to(device).detach()

            new_log_p, new_actions_index, entropy = decoder(batch_size=ppo_batch_size, old_action=actions_index)
            
            assert (actions_index == new_actions_index)
            
            reward, new_log_p, old_log_p = reward.reshape(-1), new_log_p.reshape(-1), old_log_p.reshape(-1)

            
        
            advantage = reward
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = torch.exp(new_log_p - old_log_p.detach())

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage            
            actor_loss = - torch.min(surr1, surr2).mean()

            loss = actor_loss - 0.05*entropy.mean()
            
            # print(entropy)
            # print('reward:', reward[:5].detach().cpu().numpy().tolist())
            # print('advantage:', advantage[:5].detach().cpu().numpy().tolist())
            # print('new_log_p:', new_log_p[:5].detach().cpu().numpy().tolist())
            # print('old_log_p:', old_log_p[:5].detach().cpu().numpy().tolist())
            # print('ratio:', ratio[:5].detach().cpu().numpy().tolist())
            # print('surr1:', surr1[:5].detach().cpu().numpy().tolist())
            # print('torch.min(surr1, surr2):', surr2[:5].detach().cpu().numpy().tolist())

            # print('old_log_p', old_log_p.shape)
            # print('advantage', advantage.shape)
            # print('ratio', ratio.shape)
            # print('torch.min(surr1, surr2)', torch.min(surr1, surr2).shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()
            cum_loss_counter += 1

    return cum_loss / cum_loss_counter