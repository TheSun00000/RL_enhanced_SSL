import torch
import torch.nn.functional as F

from utils.datasets import get_cifar10_dataloader
from utils.transforms import (
    get_transforms_list,
    apply_transformations
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device




def collect_trajectories(len_trajectory, encoder, decoder, batch_size, logs, neptune_run):

    assert len_trajectory % batch_size == 0

    data_loader = get_cifar10_dataloader(
        num_steps=len_trajectory // batch_size,
        batch_size=batch_size,
        transform=False
    )

    encoder_dim = encoder.projector[2].out_features


    stored_z1 = torch.zeros((len_trajectory, encoder_dim))
    stored_z2 = torch.zeros((len_trajectory, encoder_dim))
    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_transform_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_magnitude_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_rewards = torch.zeros((len_trajectory,))

    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):
#     for i in tqdm(range(len_trajectory // batch_size), desc='collect_trajectories'):

        begin, end = i*batch_size, (i+1)*batch_size

        img1, img2 = next(data_loader_iterator)

        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            _, z1 = encoder(img1)
            _, z2 = encoder(img2)

        with torch.no_grad():
            log_p, actions_index, entropies = decoder(z1, z2)
            transform_actions_index, magnitude_actions_index = actions_index
            transform_entropy, magnitude_entropy = entropies

        transforms_list_1, transforms_list_2 = get_transforms_list(transform_actions_index, magnitude_actions_index)
        new_img1 = apply_transformations(img1, transforms_list_1)
        new_img2 = apply_transformations(img2, transforms_list_2)

        new_img1 = new_img1.to(device)
        new_img2 = new_img2.to(device)
        with torch.no_grad():
            _, new_z1 = encoder(new_img1)
            _, new_z2 = encoder(new_img2)
        new_img1 = new_img1.to('cpu')
        new_img2 = new_img2.to('cpu')


        reward = - (F.normalize(new_z1) * F.normalize(new_z2)).sum(axis=-1)
        
        stored_z1[begin:end] = z1.detach().cpu()
        stored_z2[begin:end] = z1.detach().cpu()
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_transform_actions_index[begin:end]  = transform_actions_index.detach().cpu()
        stored_magnitude_actions_index[begin:end]  = magnitude_actions_index.detach().cpu()
        stored_rewards[begin:end] = reward
        
        if logs:
            neptune_run["ppo/reward"].append(reward.mean().item())
            neptune_run["ppo/transform_entropy"].append(transform_entropy.item())
            neptune_run["ppo/magnitude_entropy"].append(magnitude_entropy.item())
            
        
        
        


    return (
            (stored_z1, stored_z2), 
            (stored_log_p),
            (stored_transform_actions_index, stored_magnitude_actions_index),
            stored_rewards
        ), (img1, img2, new_img1, new_img2)


def shuffle_trajectory(trajectory):

    (
        (stored_z1, stored_z2), 
        (stored_log_p),
        (stored_transform_actions_index, stored_magnitude_actions_index),
        stored_rewards
    ) = trajectory

    permutation = torch.randperm(stored_z1.size()[0])

    permuted_stored_z1 = stored_z1[permutation]
    permuted_stored_z2 = stored_z2[permutation]
    permuted_stored_log_p = stored_log_p[permutation]
    permuted_stored_transform_actions_index  = stored_transform_actions_index[permutation]
    permuted_stored_magnitude_actions_index  = stored_magnitude_actions_index[permutation]
    permuted_stored_rewards = stored_rewards[permutation]

    permuted_trajectory = (
        (permuted_stored_z1, permuted_stored_z2),
        (permuted_stored_log_p),
        (permuted_stored_transform_actions_index, permuted_stored_magnitude_actions_index),
        permuted_stored_rewards
    )

    return permuted_trajectory


def ppo_update(trajectory, decoder, optimizer, ppo_batch_size=256, ppo_epochs=4):


    for _ in range(ppo_epochs):
    
        shuffled_trajectory = shuffle_trajectory(trajectory)

        (
            (stored_z1, stored_z2), 
            (permuted_stored_log_p),
            (stored_transform_actions_index, stored_magnitude_actions_index),
            stored_rewards
        ) = shuffled_trajectory

        len_trajectory = stored_z1.shape[0]


        assert len_trajectory % ppo_batch_size == 0

        acc_loss = 0
        acc_loss_counter = 0
        for i in range(len_trajectory // ppo_batch_size):
#         for i in tqdm(range(len_trajectory // ppo_batch_size), desc='ppo_update'):

            begin, end = i*ppo_batch_size, (i+1)*ppo_batch_size

            z1 = stored_z1[begin:end].to(device).detach()
            z2 = stored_z2[begin:end].to(device).detach()
            old_log_p = permuted_stored_log_p[begin:end].to(device).detach()
            transform_actions_index = stored_transform_actions_index[begin:end].to(device).detach()
            magnitude_actions_index = stored_magnitude_actions_index[begin:end].to(device).detach()
            reward = stored_rewards[begin:end].to(device).detach()

            new_log_p, new_actions_index, entropies = decoder(z1, z2, old_action_index=(transform_actions_index, magnitude_actions_index))
            new_transform_actions_index, new_magnitude_actions_index = new_actions_index
            
            assert (transform_actions_index == new_transform_actions_index).all()
            assert (magnitude_actions_index == new_magnitude_actions_index).all()
            
            reward, new_log_p, old_log_p = reward.reshape(-1), new_log_p.reshape(-1), old_log_p.reshape(-1)

        
            advantage = reward
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = torch.exp(new_log_p - old_log_p.detach())

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()

            loss = actor_loss

            # print('old_log_p', old_log_p.shape)
            # print('advantage', advantage.shape)
            # print('ratio', ratio.shape)
            # print('torch.min(surr1, surr2)', torch.min(surr1, surr2).shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item()
            acc_loss_counter += 1

    return acc_loss / acc_loss_counter
 