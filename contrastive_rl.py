import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from neptune.types import File
import random

from utils.datasets import get_cifar10_dataloader
from utils.networks import DecoderRNN, DecoderNoInput, build_resnet18
from utils.contrastive import InfoNCELoss, top_k_accuracy, linear_evaluation
from utils.ppo import (
    collect_trajectories_with_input,
    collect_trajectories_no_input,  
    ppo_update_with_input,
    ppo_update_no_input,
)
from utils.logs import init_neptune, get_model_save_path


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

model_save_path = get_model_save_path()
print('model_save_path:', model_save_path)


      



def ppo_init(config):

    # encoder = build_resnet18()
    # encoder.load_state_dict(torch.load('params/resnet18_contrastive_only_colorjitter.pt'))
    # encoder = encoder.to(device)

    if config['ppo_decoder']  == 'with_input':
        decoder = DecoderRNN(
            embed_size=1024,
            encoder_dim=128,
            decoder_dim=512,
            num_transforms=4,
            num_discrete_magnitude=10,
            seq_length=4
        )
    elif config['ppo_decoder']  == 'no_input':
        decoder = DecoderNoInput(
            num_transforms=4,
            num_discrete_magnitude=10,
            device=device
        )

    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=0.001
    )

    return decoder, optimizer


def contrastive_init(config):
    
    encoder = build_resnet18()
    # encoder.load_state_dict(torch.load(''))
    encoder = encoder.to(device)

    criterion = InfoNCELoss()

    optimizer = torch.optim.SGD(
        encoder.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-6,
        nesterov=True)


    def get_lr(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            100 * 50000 // 512,
            0.6,  # lr_lambda computes multiplicative factor
            1e-3
        )
    )
    
    return encoder, optimizer, scheduler, criterion


def init(config):
    
    encoder, simclr_optimizer, simclr_scheduler, simclr_criterion = contrastive_init(config)
    decoder, ppo_optimizer = ppo_init(config)
    
    return (
        (encoder, simclr_optimizer, simclr_scheduler, simclr_criterion),
        (decoder, ppo_optimizer) 
    )
    
        
def ppo_round(encoder, decoder, optimizer, config, neptune_run):
    
    ppo_rounds = config['ppo_iterations']
    len_trajectory = config['ppo_len_trajectory'] 
    batch_size = config['ppo_collection_bs'] 
    ppo_epochs = config['ppo_update_epochs'] 
    ppo_batch_size = config['ppo_update_bs']
    logs = config['logs']
    
    if config['ppo_decoder'] == 'with_input':
        collect_trajectories = collect_trajectories_with_input
        ppo_update = ppo_update_with_input
    elif config['ppo_decoder'] == 'no_input':
        collect_trajectories = collect_trajectories_no_input
        ppo_update = ppo_update_no_input
        
    
    losses = []
    rewards = []
    
    tqdm_range = tqdm(range(ppo_rounds))
    for round_ in tqdm_range:
    
        trajectory, (img1, img2, new_img1, new_img2) = collect_trajectories(
            len_trajectory=len_trajectory,
            encoder=encoder,
            decoder=decoder,
            batch_size=batch_size,
            logs=logs,
            neptune_run=neptune_run
        )

        loss = ppo_update(
            trajectory,
            decoder,
            optimizer,
            ppo_epochs=ppo_epochs,
            ppo_batch_size=ppo_batch_size,
        )
        
        losses.append(loss)
        rewards.append(float(trajectory[-1].mean()))
        
        tqdm_range.set_description(f'[ppo_round] Reward: {rewards[-1]:.4f}')
        
    
    return trajectory, (img1, img2, new_img1, new_img2), (losses, rewards)


def contrastive_round(encoder, decoder, optimizer, scheduler, criterion, random_p, config, neptune_run):
    
    num_steps = config['simclr_iterations'] 
    batch_size = config['simclr_bs']
    logs = config['logs']
    
    losses = []
    top_1_score = []
    top_5_score = []
    top_10_score = []
    
    train_loader = get_cifar10_dataloader(
        num_steps=num_steps, 
        batch_size=batch_size, 
        encoder=encoder, 
        decoder=decoder,
        random_p=random_p,
        spatial_only=False,
    )

    
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (x1, x2) in tqdm_train_loader:

        x1 = x1.to(device)
        x2 = x2.to(device)

        _, z1 = encoder(x1)
        _, z2 = encoder(x2)

        # print(x1.min(), z1.min())

        sim, _, loss = criterion(z1, z2, temperature=0.07)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if logs:
            neptune_run["simclr/loss"].append(loss.item())
            neptune_run["simclr/top_5_acc"].append(top_k_accuracy(sim, 5))
            neptune_run["simclr/sim"].append(File.as_image(sim.cpu().detach().numpy()/2 + 0.5 ))
        
        
        
        losses.append( loss.item() )
        top_1_score.append( top_k_accuracy(sim, 1) )
        top_5_score.append( top_k_accuracy(sim, 5) )
        top_10_score.append( top_k_accuracy(sim, 10) )

        tqdm_train_loader.set_description(f'[contrastive_round] Loss: {loss.item():.4f}')


        del x1, x2, loss, _
        torch.cuda.empty_cache()

#         if i % PLOT_EACH == 0:
#             clear_output(True)

#             sim, _, _ = criterion(z1, z2, batch_size=z1.shape[0], temperature=0.07)

#             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#             ax1.imshow(sim.cpu().detach(), vmin=-1, vmax=1, cmap='hot')
#             ax2.plot(smooth_curve(losses))
#             ax3.plot(smooth_curve(top_1_score))
#             ax3.plot(smooth_curve(top_5_score))
#             ax3.plot(smooth_curve(top_10_score))

#             plt.show()
#         break
        
    return (sim.cpu().detach(), losses, top_1_score, top_5_score, top_10_score)


# config = {
#     'iterations':100,
    
#     'simclr_iterations':50,
#     'simclr_bs':1024,
#     'linear_eval_epochs':100,
    
#     'ppo_decoder': 'no_input', # ['no_input', 'with_input']
#     'ppo_iterations':100,
#     'ppo_len_trajectory':512*4,
#     'ppo_collection_bs':512*2,
#     'ppo_update_bs':256,
#     'ppo_update_epochs':4,
    
#     'logs':True,
#     'model_save_path':model_save_path,
#     'seed':seed,
    
# }


config = {
    'iterations':100,
    
    'simclr_iterations':50,
    'simclr_bs':1024,
    'linear_eval_epochs':100,
    
    'ppo_decoder': 'no_input', # ['no_input', 'with_input']
    'ppo_iterations':50,
    'ppo_len_trajectory':512,
    'ppo_collection_bs':512,
    'ppo_update_bs':256,
    'ppo_update_epochs':4,
    
    'logs':True,
    'model_save_path':model_save_path,
    'seed':seed,
    
}



(
    (encoder, simclr_optimizer, simclr_scheduler, simclr_criterion),
    (decoder, ppo_optimizer) 
) = init(config)


ppo_rewards_metric = []
crst_losses = []
crst_top_1_score = []
crst_top_5_score = []
crst_top_10_score = []


logs = config['logs']
neptune_run = init_neptune(['contrastive_rl'] + [f'{k}={v}' for (k, v) in config.items()]) if logs else None

if logs:
    neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])

for step in tqdm(range(config['iterations']), desc='[Main Loop]'):
        
    (sim, losses, top_1_score, top_5_score, top_10_score) = contrastive_round(
        encoder,
        decoder,
        config=config,
        optimizer=simclr_optimizer, 
        scheduler=simclr_scheduler, 
        criterion=simclr_criterion, 
        random_p=1 if step == 0 else 0.5,
        # random_p=1,
        neptune_run=neptune_run
    )
    crst_losses += losses
    crst_top_1_score += top_1_score
    crst_top_5_score += top_5_score
    crst_top_10_score += top_10_score


    trajectory, (img1, img2, new_img1, new_img2), (ppo_losses, ppo_rewards) = ppo_round(
        encoder, 
        decoder,
        ppo_optimizer,
        config=config,
        neptune_run=neptune_run
    )
    ppo_rewards_metric += ppo_rewards
    
    
    
    torch.save(encoder.state_dict(), f'{model_save_path}/encoder.pt')
    torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/encoder_opt.pt')
    torch.save(simclr_scheduler.state_dict(), f'{model_save_path}/encoder_shd.pt')
    torch.save(decoder.state_dict(), f'{model_save_path}/decoder.pt')
    torch.save(ppo_optimizer.state_dict(), f'{model_save_path}/decoder_opt.pt')
    
    
    if logs:
        neptune_run["params/encoder"].upload(f'{model_save_path}/encoder.pt')
        neptune_run["params/encoder_opt"].upload(f'{model_save_path}/encoder_opt.pt')
        neptune_run["params/encoder_shd"].upload(f'{model_save_path}/encoder_shd.pt')
        neptune_run["params/decoder"].upload(f'{model_save_path}/decoder.pt')
        neptune_run["params/decoder_opt"].upload(f'{model_save_path}/decoder_opt.pt')
    
    
    train_acc, test_acc = linear_evaluation(encoder, num_epochs=config['linear_eval_epochs'])
    
    if logs:
        neptune_run["linear_eval/train_acc"].append(train_acc)
        neptune_run["linear_eval/test_acc"] .append(test_acc)
    

if logs:
    neptune_run.stop()