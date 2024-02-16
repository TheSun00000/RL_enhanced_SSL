import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter



from utils.datasets import (
    get_cifar10_dataloader, 
    rotate_images, 
    plot_images_stacked,
    select_from_rotated_views   
)
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


def distance_between_positions(position1, position2):
    ax, ay = position1 // 3, position1 % 3
    bx, by = position2 // 3, position2 % 3
    distance = ((ax-bx)**2 + (ay-by)**2)**0.5
    # 2.8284 is the max distance ( the diagonal )
    return distance / 2.8284


def get_transformations_strength(actions):
    
    batch_size = actions.shape[0]
    
    crop_position1 = actions[..., 0, 0]
    crop_position2 = actions[..., 1, 0]
    position_strength = distance_between_positions(crop_position1, crop_position2).float().reshape(batch_size, -1)
    area_strength = (10 - actions[..., 1].float()).reshape(batch_size, -1) / 10
    color_jitter_strength = actions[..., 2:6].float().reshape(batch_size, -1) / 10
    
    # print(position_strength.mean(dim=1))
    # print(area_strength.mean(dim=1))
    # print(color_jitter_strength.mean(dim=1))
    
    transformations_strength = position_strength.mean(dim=1) + area_strength.mean(dim=1) + color_jitter_strength.mean(dim=1)
    return transformations_strength / 3


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
    
    normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    encoder_dim = encoder.projector[3].out_features


    stored_z = torch.zeros((len_trajectory, encoder_dim))
    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_actions_index  = torch.zeros((len_trajectory, 2, 10), dtype=torch.long)
    stored_rewards = torch.zeros((len_trajectory,))

    
    mean_rewards = 0
    mean_rot_reward = 0
    mean_strength = 0
    mean_entropy = 0    
    
    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):
#     for i in tqdm(range(len_trajectory // batch_size), desc='collect_trajectories'):

        begin, end = i*batch_size, (i+1)*batch_size

        img, y = next(data_loader_iterator)
        normalized_img = torch.stack([normalization(tensor) for tensor in img])
        normalized_img = normalized_img.to(device)

        with torch.no_grad():
            _, z = encoder(normalized_img)
            log_p, actions_index, entropy = decoder(z)

        num_discrete_magnitude = decoder.num_discrete_magnitude
        transforms_list_1, transforms_list_2 = get_transforms_list(
            actions_index,
            num_magnitudes=num_discrete_magnitude)
        new_img1 = apply_transformations(img, transforms_list_1)
        new_img2 = apply_transformations(img, transforms_list_2)

        new_img1 = torch.stack([normalization(tensor) for tensor in new_img1])
        new_img2 = torch.stack([normalization(tensor) for tensor in new_img2])
        
        
        new_img1 = new_img1.to(device)
        new_img2 = new_img2.to(device)
        
        with torch.no_grad():
            # _, new_z1 = encoder(new_img1)
            # _, new_z2 = encoder(new_img2)
            
            rotated_x1, rotated_labels1 = rotate_images(new_img1)
            rotated_x2, rotated_labels2 = rotate_images(new_img2)
            
            rotated_x, rotated_labels = select_from_rotated_views(
                rotated_x1, rotated_x2,
                rotated_labels1, rotated_labels2
            )            
            
            # print(rotated_labels[:10])
            # plot_images_stacked(rotated_x[:5], rotated_x[5:10])
            
            rotated_x = rotated_x.to(device)
            feature = encoder.enc(rotated_x)
            logits = encoder.predictor(feature)
            
            # rot_reward = F.cross_entropy(logits, rotated_labels, reduce=False)
            # rot_reward = rot_reward.reshape(-1, 4).mean(dim=-1)
            
            predicttion = logits.argmax(dim=-1)
            rot_acc = 1. * (predicttion == rotated_labels)
            rot_acc = rot_acc.reshape(-1, 4).mean(dim=-1)
                        
            
        new_img1 = new_img1.to('cpu')
        new_img2 = new_img2.to('cpu')

        transformations_strength = get_transformations_strength(actions_index)
        # reward = similariy_reward_function(new_z1, new_z2) + rot_reward
        
        # print(transformations_strength)
        reward = transformations_strength + rot_acc
        
        stored_z[begin:end] = z.detach().cpu()
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_actions_index[begin:end]  = actions_index.detach().cpu()
        stored_rewards[begin:end] = reward
        
        mean_rewards += reward.mean().item()
        mean_rot_reward += rot_acc.mean().item()
        mean_strength += transformations_strength.mean().item()
        mean_entropy += entropy.item()

    
    # string_transforms = []
    # for trans in transforms_list_1:
    #     s = ' '.join([ f'{name}_{magnetude}' for (name, _, magnetude) in trans])
    #     string_transforms.append( s )
    # print_sorted_strings_with_counts(string_transforms, topk=5)
    
    mean_rewards /= (len_trajectory // batch_size)
    mean_rot_reward /= (len_trajectory // batch_size)
    mean_strength /= (len_trajectory // batch_size)
    mean_entropy /= (len_trajectory // batch_size)
    
    if logs:
        neptune_run["ppo/reward"].append(mean_rewards)
        neptune_run["ppo/rot_reward"].append(mean_rot_reward)
        neptune_run["ppo/strength_reward"].append(mean_strength)
        neptune_run["ppo/mean_entropy"].append(mean_entropy)

    return (
            (stored_z), 
            (stored_log_p),
            (stored_actions_index),
            stored_rewards
        ), (img, img, new_img1, new_img2), entropy


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


def shuffle_trajectory_2(trajectory):

    (
        (stored_z), 
        (stored_log_p),
        (actions_index),
        stored_rewards
    ) = trajectory

    permutation = torch.randperm(stored_z.shape[0])

    permuted_stored_z = stored_z[permutation]
    permuted_stored_log_p = stored_log_p[permutation]
    permuted_stored_actions_index  = actions_index[permutation]
    permuted_stored_rewards = stored_rewards[permutation]

    permuted_trajectory = (
        (permuted_stored_z),
        (permuted_stored_log_p),
        (permuted_stored_actions_index),
        permuted_stored_rewards
    )

    return permuted_trajectory


def ppo_update_with_input(trajectory, decoder, optimizer, ppo_batch_size=256, ppo_epochs=4):


    for _ in range(ppo_epochs):
    
        shuffled_trajectory = shuffle_trajectory_2(trajectory)

        (
            (stored_z), 
            (permuted_stored_log_p),
            (stored_actions_index),
            stored_rewards
        ) = shuffled_trajectory
        
        len_trajectory = stored_z.shape[0]


        assert len_trajectory % ppo_batch_size == 0

        acc_loss = 0
        acc_loss_counter = 0
        for i in range(len_trajectory // ppo_batch_size):
#         for i in tqdm(range(len_trajectory // ppo_batch_size), desc='ppo_update'):

            begin, end = i*ppo_batch_size, (i+1)*ppo_batch_size

            z = stored_z[begin:end].to(device).detach()
            old_log_p = permuted_stored_log_p[begin:end].to(device).detach()
            actions_index = stored_actions_index[begin:end].to(device).detach()
            reward = stored_rewards[begin:end].to(device).detach()

            new_log_p, new_actions_index, entropy = decoder(z, old_action_index=actions_index)
            
            assert (actions_index == new_actions_index).all()
            
            reward, new_log_p, old_log_p = reward.reshape(-1), new_log_p.reshape(-1), old_log_p.reshape(-1)

            
        
            advantage = reward
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = torch.exp(new_log_p - old_log_p.detach())

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage            
            actor_loss = - torch.min(surr1, surr2).mean()

            loss = actor_loss
            
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
 