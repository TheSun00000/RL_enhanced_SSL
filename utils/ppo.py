import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter



from utils.datasets import get_cifar10_dataloader
from utils.transforms import (
    get_transforms_list,
    apply_transformations
)
from utils.contrastive import InfoNCELoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device

# random_grayscale = transforms.RandomGrayscale(p=1)

infonce_reward = InfoNCELoss(reduction='none')

def similariy_reward_function(new_z1, new_z2):
    return - (F.normalize(new_z1) * F.normalize(new_z2)).sum(axis=-1)

def infonce_reward_function(new_z1, new_z2):
    bs = new_z1.shape[0]
    full_similarity_matrix, logits, loss = infonce_reward(new_z1, new_z2, temperature=0.5)
    reward = (loss[:bs] + loss[bs:]) / 2
    return reward

def test_reward_function(y, magnitude_actions_index):
    # print(y.shape, magnitude_actions_index.shape)
    
    reward = (y.reshape(-1,1,1).to(device) == magnitude_actions_index)
    reward = reward.reshape(y.shape[0], -1)
    reward = reward.sum(dim=-1)

    return reward * 1.


def print_sorted_strings_with_counts(input_list, topk):
    # Count occurrences of each string
    string_counts = Counter(input_list)

    # Sort strings by counts in descending order
    sorted_strings = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)[:topk]

    # Print the sorted strings with counts
    for i, (string, count) in enumerate(sorted_strings):
        print(f"{i}) {string}: {count}")


def collect_trajectories_with_input(len_trajectory, encoder, decoder, batch_size, logs, neptune_run):

    assert len_trajectory % batch_size == 0

    data_loader = get_cifar10_dataloader(
        num_steps=len_trajectory // batch_size,
        batch_size=batch_size,
        spatial_only=True,
    )

    encoder_dim = encoder.projector[2].out_features


    stored_z1 = torch.zeros((len_trajectory, encoder_dim))
    stored_z2 = torch.zeros((len_trajectory, encoder_dim))
    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_transform_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_magnitude_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_rewards = torch.zeros((len_trajectory,))

    
    mean_rewards = 0
    mean_transform_entropy = 0
    mean_magnitude_entropy = 0
    
    
    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):
#     for i in tqdm(range(len_trajectory // batch_size), desc='collect_trajectories'):

        begin, end = i*batch_size, (i+1)*batch_size

        (img1, img2), y = next(data_loader_iterator)

        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            _, z1 = encoder(img1)
            _, z2 = encoder(img2)
            log_p, actions_index, entropies = decoder(z1, z2)
            transform_actions_index, magnitude_actions_index = actions_index
            transform_entropy, magnitude_entropy = entropies

        num_discrete_magnitude = decoder.num_discrete_magnitude
        transforms_list_1, transforms_list_2 = get_transforms_list(
            transform_actions_index,
            magnitude_actions_index,
            num_magnitudes=num_discrete_magnitude)
        new_img1 = apply_transformations(img1, transforms_list_1)
        new_img2 = apply_transformations(img2, transforms_list_2)

        # new_img1 = torch.stack([random_grayscale(tensor) for tensor in new_img1])
        # new_img2 = torch.stack([random_grayscale(tensor) for tensor in new_img2])
        
        new_img1 = new_img1.to(device)
        new_img2 = new_img2.to(device)
        with torch.no_grad():
            _, new_z1 = encoder(new_img1)
            _, new_z2 = encoder(new_img2)
        new_img1 = new_img1.to('cpu')
        new_img2 = new_img2.to('cpu')


        reward = similariy_reward_function(new_z1, new_z2)
        # reward = test_reward_function(y, magnitude_actions_index)
        
        stored_z1[begin:end] = z1.detach().cpu()
        stored_z2[begin:end] = z1.detach().cpu()
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_transform_actions_index[begin:end]  = transform_actions_index.detach().cpu()
        stored_magnitude_actions_index[begin:end]  = magnitude_actions_index.detach().cpu()
        stored_rewards[begin:end] = reward
        
        mean_rewards += reward.mean().item()
        mean_transform_entropy += transform_entropy.item()
        mean_magnitude_entropy += magnitude_entropy.item()

    
    string_transforms = []
    for trans in transforms_list_1:
        s = ' '.join([ f'{name}_{magnetude}' for (name, _, magnetude) in trans])
        string_transforms.append( s )
    # print_sorted_strings_with_counts(string_transforms, topk=5)
    
    mean_rewards /= (len_trajectory // batch_size)
    mean_transform_entropy /= (len_trajectory // batch_size)
    mean_magnitude_entropy /= (len_trajectory // batch_size)
    
    entropy = (mean_transform_entropy + mean_magnitude_entropy) / 2

    if logs:
        neptune_run["ppo/reward"].append(mean_rewards)
        neptune_run["ppo/transform_entropy"].append(mean_transform_entropy)
        neptune_run["ppo/magnitude_entropy"].append(mean_magnitude_entropy)

    return (
            (stored_z1, stored_z2), 
            (stored_log_p),
            (stored_transform_actions_index, stored_magnitude_actions_index),
            stored_rewards
        ), (img1, img2, new_img1, new_img2), entropy


def collect_trajectories_no_input(len_trajectory, encoder, decoder, batch_size, logs, neptune_run):

    assert len_trajectory % batch_size == 0

    data_loader = get_cifar10_dataloader(
        num_steps=len_trajectory // batch_size,
        batch_size=batch_size,
        spatial_only=True,
    )

    encoder_dim = encoder.projector[2].out_features


    stored_z1 = torch.zeros((len_trajectory, encoder_dim))
    stored_z2 = torch.zeros((len_trajectory, encoder_dim))
    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_transform_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_magnitude_actions_index  = torch.zeros((len_trajectory, 2, decoder.seq_length), dtype=torch.long)
    stored_rewards = torch.zeros((len_trajectory,))

    mean_rewards = 0
    mean_transform_entropy = 0
    mean_magnitude_entropy = 0
    
    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):
#     for i in tqdm(range(len_trajectory // batch_size), desc='collect_trajectories'):

        begin, end = i*batch_size, (i+1)*batch_size

        (img1, img2), y = next(data_loader_iterator)

        with torch.no_grad():
            log_p, actions_index, entropies = decoder(batch_size)
            transform_actions_index, magnitude_actions_index = actions_index
            transform_entropy, magnitude_entropy = entropies
            
            new_log_p, new_actions_index, new_entropies = decoder(batch_size, old_action_index=actions_index)
            new_transform_actions_index, new_magnitude_actions_index = new_actions_index
            new_transform_entropy, new_magnitude_entropy = new_entropies
            
            assert (log_p == new_log_p).all(), "haya yerham babak"
            assert (transform_actions_index == new_transform_actions_index).all(), "haya yerham babak"
            assert (magnitude_actions_index == new_magnitude_actions_index).all(), "haya yerham babak"
            assert (transform_entropy == new_transform_entropy).all(), "haya yerham babak"
            assert (magnitude_entropy == new_magnitude_entropy).all(), "haya yerham babak"
            
            

        num_discrete_magnitude = decoder.num_discrete_magnitude
        transforms_list_1, transforms_list_2 = get_transforms_list(
            transform_actions_index,
            magnitude_actions_index,
            num_magnitudes=num_discrete_magnitude)
        
        
        
        new_img1 = apply_transformations(img1, transforms_list_1)
        new_img2 = apply_transformations(img2, transforms_list_2)
        
        # new_img1 = torch.stack([random_grayscale(tensor) for tensor in new_img1])
        # new_img2 = torch.stack([random_grayscale(tensor) for tensor in new_img2])

        new_img1 = new_img1.to(device)
        new_img2 = new_img2.to(device)
        with torch.no_grad():
            _, new_z1 = encoder(new_img1)
            _, new_z2 = encoder(new_img2)
        new_img1 = new_img1.to('cpu')
        new_img2 = new_img2.to('cpu')


        # reward = similariy_reward_function(new_z1, new_z2)
        reward = infonce_reward_function(new_z1, new_z2)
                
        stored_z1[begin:end] = new_z1.detach().cpu()
        stored_z2[begin:end] = new_z2.detach().cpu()
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_transform_actions_index[begin:end]  = transform_actions_index.detach().cpu()
        stored_magnitude_actions_index[begin:end]  = magnitude_actions_index.detach().cpu()
        stored_rewards[begin:end] = reward
            
        mean_rewards += reward.mean().item()
        mean_transform_entropy += transform_entropy.item()
        mean_magnitude_entropy += magnitude_entropy.item()
    
    
    # transforms_list_1, transforms_list_2 = get_transforms_list(
    #     transform_actions_index, 
    #     magnitude_actions_index,
    #     num_magnitudes=decoder.num_discrete_magnitude
    # )
    # string_transforms = []
    # for trans in transforms_list_1:
    #     s = ' '.join([ f'{name}({magnetude})' for (name, _, magnetude) in trans])
    #     string_transforms.append( s )
    # print_sorted_strings_with_counts(string_transforms, topk=5)
    
    
    mean_rewards /= (len_trajectory // batch_size)
    mean_transform_entropy /= (len_trajectory // batch_size)
    mean_magnitude_entropy /= (len_trajectory // batch_size)
    
    entropy = (mean_transform_entropy + mean_magnitude_entropy) / 2

    if logs:
        neptune_run["ppo/reward"].append(mean_rewards)
        neptune_run["ppo/transform_entropy"].append(mean_transform_entropy)
        neptune_run["ppo/magnitude_entropy"].append(mean_magnitude_entropy)

    return (
            (stored_z1, stored_z2), 
            (stored_log_p),
            (stored_transform_actions_index, stored_magnitude_actions_index),
            stored_rewards
        ), (img1, img2, new_img1, new_img2), entropy



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


def ppo_update_with_input(trajectory, decoder, optimizer, ppo_batch_size=256, ppo_epochs=4):


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



def ppo_update_no_input(trajectory, decoder, optimizer, ppo_batch_size=256, ppo_epochs=4):


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

            old_log_p = permuted_stored_log_p[begin:end].to(device).detach()
            transform_actions_index = stored_transform_actions_index[begin:end].to(device).detach()
            magnitude_actions_index = stored_magnitude_actions_index[begin:end].to(device).detach()
            reward = stored_rewards[begin:end].to(device).detach()

            new_log_p, new_actions_index, entropies = decoder(ppo_batch_size, old_action_index=(transform_actions_index, magnitude_actions_index))
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

            entropy_loss = (entropies[0] + entropies[1])/2
            
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
 