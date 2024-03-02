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
    apply_transformations,
    NUM_DISCREATE
)
from utils.contrastive import InfoNCELoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device

# random_grayscale = transforms.RandomGrayscale(p=1)

infonce_reward = InfoNCELoss(reduction='none')


def standarize_reward(reward, min, max):
    return (reward - min)/(max-min)

def similariy_reward_function(new_z1, new_z2):
    return - (F.normalize(new_z1) * F.normalize(new_z2)).sum(axis=-1)

def infonce_reward_function(new_z1, new_z2):
    bs = new_z1.shape[0]
    full_similarity_matrix, logits, loss = infonce_reward(new_z1, new_z2, temperature=0.5)
    reward = loss
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
    
    batch_size = len(actions)
    
    color_jitter_strength = []
    for b in range(len(actions)):
        color_jitter_strength.append([])
        for i in range(len(actions[0])):
            mean_level = sum([level for name, pr, level in actions[b][i]]) / len(actions[0][0])
            color_jitter_strength[-1].append(mean_level)
    
    color_jitter_strength = torch.tensor(color_jitter_strength)
    return color_jitter_strength


def print_sorted_strings_with_counts(input_list, topk):
    # Count occurrences of each string
    string_counts = Counter(input_list)

    # Sort strings by counts in descending order
    sorted_strings = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)[:topk]

    # Print the sorted strings with counts
    for i, (string, count) in enumerate(sorted_strings):
        print(f"{i}) {string}: {count}")


# #########################################################################################################################


def collect_trajectories_with_input(len_trajectory, encoder, decoder, batch_size, max_strength, logs, neptune_run):

    assert len_trajectory % batch_size == 0

    data_loader = get_cifar10_dataloader(
        num_steps=len_trajectory // batch_size,
        batch_size=batch_size,
        spatial_only=True,
    )
    
    normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    last_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    # encoder_dim = encoder.projector[3].out_features
    encoder_dim = 2048


    stored_log_p = torch.zeros((len_trajectory, 1))
    stored_actions_index = []
    stored_rewards = torch.zeros((len_trajectory,))

    
    mean_rewards = 0
    mean_rot_reward = 0
    mean_strength = 0
    mean_entropy = 0
    mean_infonce_reward = 0 
    
    data_loader_iterator = iter(data_loader)
    for i in range(len_trajectory // batch_size):
#     for i in tqdm(range(len_trajectory // batch_size), desc='collect_trajectories'):

        begin, end = i*batch_size, (i+1)*batch_size

        img, y = next(data_loader_iterator)

        with torch.no_grad():
            log_p, actions_index, entropy = decoder(batch_size=batch_size)
            new_log_p, new_actions_index, new_entropy = decoder(batch_size=batch_size, old_action=actions_index)
            assert (log_p == new_log_p).all()
            assert (actions_index == new_actions_index)
            assert (entropy == new_entropy).all()
            
        num_discrete_magnitude = decoder.num_discrete_magnitude
        transforms_list_1, transforms_list_2 = get_transforms_list(
            actions_index,
            num_magnitudes=num_discrete_magnitude)
        

        new_img1 = apply_transformations(img, transforms_list_1, max_strength)
        new_img2 = apply_transformations(img, transforms_list_2, max_strength)

        new_img1 = torch.stack([last_transform(img) for img in new_img1])
        new_img2 = torch.stack([last_transform(img) for img in new_img2])
        
        
        new_img1 = new_img1.to(device)
        new_img2 = new_img2.to(device)
        
        rotation_reward = None
        infoNCE_reward = None
        transformations_strength = None
        
        with torch.no_grad():
            _, new_z1 = encoder(new_img1)
            _, new_z2 = encoder(new_img2)
            
        #     rotated_x1, rotated_labels1 = rotate_images(new_img1)
        #     rotated_x2, rotated_labels2 = rotate_images(new_img2)
            
        #     rotated_x, rotated_labels = select_from_rotated_views(
        #         rotated_x1, rotated_x2,
        #         rotated_labels1, rotated_labels2
        #     )            
            
        #     # print(rotated_labels[:10])
        #     # plot_images_stacked(rotated_x[:5], rotated_x[5:10])
            
        #     rotated_x = rotated_x.to(device)
        #     rotated_labels = rotated_labels.to(device)
            
        #     feature = encoder.enc(rotated_x)
        #     logits = encoder.predictor2(feature)
            
        #     rot_loss = F.cross_entropy(logits, rotated_labels, reduce=False)
        #     rot_loss = rot_loss.reshape(-1, 4).mean(dim=-1)
            
        #     predicttion = logits.argmax(dim=-1)
        #     rot_acc = 1. * (predicttion == rotated_labels)
        #     rot_acc = rot_acc.reshape(-1, 4).mean(dim=-1)
                        
            
        new_img1 = new_img1.to('cpu')
        new_img2 = new_img2.to('cpu')

        transformations_strength = get_transformations_strength(actions_index)
        # reward = transformations_strength + rot_acc
        # reward = similariy_reward_function(new_z1, new_z2) + rot_acc
        
        # infoNCE_reward = infonce_reward_function(new_z1, new_z2)
        # infoNCE_reward = standarize_reward(infoNCE_reward, 4.5, 5.5)
        
        # rotation_reward = 1 - standarize_reward(rot_loss, 0, 1.5)
        # reward = infoNCE_reward + rotation_reward
        # reward=infoNCE_reward
        reward = similariy_reward_function(new_z1, new_z2)
        
        stored_log_p[begin:end] = log_p.detach().cpu()
        stored_actions_index += actions_index
        stored_rewards[begin:end] = reward
        
        mean_rewards += reward.mean().item()
        mean_entropy += entropy.item()
        
        if rotation_reward is not None:
            mean_rot_reward += rotation_reward.mean().item()
        if infoNCE_reward is not None:
            mean_infonce_reward += infoNCE_reward.mean().item()
        if transformations_strength is not None:
            mean_strength += transformations_strength.mean().item()

    
    string_transforms = []
    for trans1, trans2 in zip(transforms_list_1, transforms_list_2):
        s1 = ' '.join([ f'{name[:4]}_{round(level, 3)}' for (name, _, level) in trans1])
        s2 = ' '.join([ f'{name[:4]}_{round(level, 3)}' for (name, _, level) in trans2])
        string_transforms.append( f'{s1}  ||  {s2}' )
    print_sorted_strings_with_counts(string_transforms, topk=5)
    
    # plot_images_stacked(new_img1[:10], new_img2[:10])
    
    mean_rewards /= (len_trajectory // batch_size)
    mean_rot_reward /= (len_trajectory // batch_size)
    mean_infonce_reward /= (len_trajectory // batch_size)
    mean_strength /= (len_trajectory // batch_size)
    mean_entropy /= (len_trajectory // batch_size)
    
    if logs:
        neptune_run["ppo/reward"].append(mean_rewards)
        neptune_run["ppo/mean_entropy"].append(mean_entropy)
        if rotation_reward is not None:
            neptune_run["ppo/rot_reward"].append(mean_rot_reward)
        if infoNCE_reward is not None:
            neptune_run["ppo/infonce_reward"].append(mean_infonce_reward)
        if transformations_strength is not None:
            neptune_run["ppo/strength_reward"].append(mean_strength)

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


def ppo_update_with_input(trajectory, decoder, optimizer, ppo_batch_size=256, ppo_epochs=4):


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
            
            cum_loss += loss.item()
            cum_loss_counter += 1

    return cum_loss / cum_loss_counter