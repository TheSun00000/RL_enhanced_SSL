import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from neptune.types import File
import random
import math
import neptune

from utils.datasets import (
    get_cifar10_dataloader, 
    rotate_images, 
    plot_images_stacked,
    select_from_rotated_views
)

from utils.networks import SimCLR, DecoderNN_1input, build_resnet18, build_resnet50
from utils.contrastive import InfoNCELoss, top_k_accuracy, knn_evaluation, info_nce_loss, knn_monitor
from utils.ppo import (
    collect_trajectories_with_input,
    ppo_update_with_input,
    print_sorted_strings_with_counts
)
from utils.transforms import get_transforms_list, NUM_DISCREATE, transformations_dict
from utils.logs import init_neptune, get_model_save_path
from utils.resnet import resnet18

# seed = random.randint(0, 100000)
seed = 0
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


def cuda_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a
    
    print(f'total memory: {t/1024**3:.4f}')
    print(f'reserved memory: {r/1024**3:.4f}')
    print(f'allocated memory: {a/1024**3:.4f}')
    print(f'free memory: {f/1024**3:.4f}')
    



def ppo_init(config):

    # encoder = build_resnet18()
    # encoder.load_state_dict(torch.load('params/resnet18_contrastive_only_colorjitter.pt'))
    # encoder = encoder.to(device)

    if config['ppo_decoder']  == 'with_input':
        decoder = DecoderNN_1input(
            transforms=list(transformations_dict.keys()),
            num_discrete_magnitude=NUM_DISCREATE,
            device=device
        )
    else:
        raise NotImplementedError
    
    # decoder.load_state_dict(torch.load('params/params_125/decoder.pt'))
    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=0.01
    )
    
    # if config['checkpoint_params']:
    #     checkpoint_params = config['checkpoint_params']
    #     decoder.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder.pt'))
    #     optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder_opt.pt'))

    return decoder, optimizer


def contrastive_init(config):
    
    if config['encoder_backbone'] == 'resnet18':
        encoder = build_resnet18()
    elif config['encoder_backbone'] == 'resnet50':
        encoder = build_resnet50()
    
    
    # encoder.load_state_dict(torch.load('params/params_222/encoder.pt'))
    encoder = encoder.to(device)

    criterion = InfoNCELoss()

    # # # # # # # optimizer = torch.optim.SGD(
    # # # # # # #     encoder.parameters(),
    # # # # # # #     lr=0.01,
    # # # # # # #     momentum=0.9,
    # # # # # # #     weight_decay=1e-6,
    # # # # # # #     nesterov=True)
    
    # optimizer = torch.optim.Adam(
    #     encoder.parameters(),
    #     lr=0.001,
    #     weight_decay=1e-6,
    # )
    
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        momentum=0.9,
        lr=config['lr'] * config['simclr_bs'] / 256,
        weight_decay=0.0005
    )


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
    
    if config['checkpoint_params']:
        checkpoint_params = config['checkpoint_params']
        encoder.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder.pt'))
        optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder_opt.pt'))
    
    return encoder, optimizer, scheduler, criterion


def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
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


def init(config):
    
    encoder, simclr_optimizer, simclr_scheduler, simclr_criterion = contrastive_init(config)
    decoder, ppo_optimizer = ppo_init(config)
    
    return (
        (encoder, simclr_optimizer, simclr_scheduler, simclr_criterion),
        (decoder, ppo_optimizer) 
    )
    
        
def ppo_round(encoder, decoder, optimizer, max_strength, config, neptune_run):
    
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
    
    tqdm_range = tqdm(range(ppo_rounds), desc='[ppo_round]')
    for round_ in tqdm_range:
    
        trajectory, (img1, img2, new_img1, new_img2), entropy = collect_trajectories(
            len_trajectory=len_trajectory,
            encoder=encoder,
            decoder=decoder,
            batch_size=batch_size,
            max_strength=max_strength,
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
        
        # tqdm_range.set_description(f'[ppo_round] Reward: {rewards[-1]:.4f}')
        
    stored_actions_index = trajectory[2]
    transforms_list_1, transforms_list_2 = get_transforms_list(
        stored_actions_index,
        num_magnitudes=decoder.num_discrete_magnitude
    )
    string_transforms = []
    for trans1, trans2 in zip(transforms_list_1, transforms_list_2):
        s1 = ' '.join([ f'{name[:4]}_{magnetude}' for (name, _, magnetude) in trans1])
        s2 = ' '.join([ f'{name[:4]}_{magnetude}' for (name, _, magnetude) in trans2])
        string_transforms.append( f'{s1}  ||  {s2}' )
    print_sorted_strings_with_counts(string_transforms, topk=5)
        
    
    return trajectory, (img1, img2, new_img1, new_img2), entropy, (losses, rewards)


def contrastive_round(encoder: SimCLR, decoder, optimizer, max_strength, scheduler, criterion, random_p, config, epoch, neptune_run):
    
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
    
    # train_loader = get_essl_train_loader()

    
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader), desc='[contrastive_round]')
    
    lr = None
    
    encoder.train()
    
    for it, ((x1, x2), y) in tqdm_train_loader:
        
        lr = adjust_learning_rate(epochs=config['epochs'],
            warmup_epochs=config['warmup_epochs'],
            base_lr=config['lr'] * config['simclr_bs'] / 256,
            optimizer=optimizer,
            loader=train_loader,
            step=it+(epoch-1)*len(train_loader)
        )
        
        _, z1 = encoder(x1.to(device))
        _, z2 = encoder(x2.to(device))

        sim, _, simclr_loss = criterion(z1, z2, temperature=0.5)

        simclr_loss_item = simclr_loss.item()
        loss = simclr_loss

        
        if config['lmbd'] > 0:            
            rotated_x1, rotated_labels1 = rotate_images(x1)
            rotated_x2, rotated_labels2 = rotate_images(x2)
                        
            rotated_x, rotated_labels = select_from_rotated_views(
                rotated_x1, rotated_x2,
                rotated_labels1, rotated_labels2
            )
                             
            rotated_x = rotated_x.to(device)
            rotated_labels = rotated_labels.to(device)
            
            feature = encoder.enc(rotated_x)
            logits = encoder.predictor2(feature)
            rot_loss = F.cross_entropy(logits, rotated_labels)
            loss += config['lmbd'] * rot_loss
            
            rot_acc = (logits.argmax(dim=-1) == rotated_labels).sum() / len(rotated_labels)
                            
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logs:
            neptune_run["simclr/loss"].append(simclr_loss_item)
            if config['lmbd'] > 0:
                neptune_run["simclr/all_loss"].append(loss.item())
                neptune_run["simclr/rot_loss"].append(rot_loss.item())
                neptune_run["simclr/rot_acc"].append(rot_acc.item())
            # neptune_run["simclr/top_5_acc"].append(top_k_accuracy(sim, 5))
            # neptune_run["simclr/top_1_acc"].append(top_k_accuracy(sim, 1))
        
        
        
        losses.append( loss.item() )
        # top_1_score.append( top_k_accuracy(sim, 1) )
        # top_5_score.append( top_k_accuracy(sim, 5) )
        # top_10_score.append( top_k_accuracy(sim, 10) )

        # tqdm_train_loader.set_description(f'[contrastive_round] Loss: {loss.item():.4f}')


        del x1, x2, loss, _
        torch.cuda.empty_cache()
    
    if lr:
        print('step:{}   lr:{}'.format(it+(epoch-1)*len(train_loader), lr))
    
    # return (sim.cpu().detach(), losses, top_1_score, top_5_score, top_10_score)

       

def get_random_p(epoch, init_random_p):
    return 1 - (1 - min(epoch, 40)/40)*init_random_p


config = {
    'epochs':800,
    'warmup_epochs':10,

    'simclr_iterations':'all',
    'simclr_bs':512,
    'linear_eval_epochs':200,
    'random_p':0.0,
    'encoder_backbone': 'resnet18', # ['resnet18', 'resnet50']
    'lmbd': 0.0,
    'lr':0.03,
    
    'ppo_decoder': 'with_input', # ['no_input', 'with_input']
    'ppo_iterations':200,
    'ppo_len_trajectory':128,
    'ppo_collection_bs':128,
    'ppo_update_bs':16,
    'ppo_update_epochs':4,
    'max_strength':1,
    
    'logs':True,
    'model_save_path':model_save_path,
    'seed':seed,
    
    'checkpoint_id':"",
    'checkpoint_params':"",
    # 'checkpoint_id':"SIM-323",
    # 'checkpoint_params':'params_433',
    
}


(
    (encoder, simclr_optimizer, simclr_scheduler, simclr_criterion),
    (decoder, ppo_optimizer) 
) = init(config)



logs = config['logs']
logs_keys = ['random_p', 'max_strength', 'ppo_iterations']
neptune_run = init_neptune(['contrastive_rl'] + [f'{k}={config[k]}' for (k) in logs_keys]) if logs else None

if logs:
    neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])

stop_ppo = False


start_epoch = 1

if config['checkpoint_params']:
    
    prev_run = neptune.init_run(
        project="nazim-bendib/simclr-rl",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
        with_id=config['checkpoint_id']
    )
    
    # [tag.split('=')[1] for tag in list(run['sys/tags'].fetch()) if 'model_save_path=' in tag][0]
    
    test_acc = prev_run['linear_eval/test_acc'].fetch_values().value.tolist()
    start_epoch = len(test_acc) + 1
    
    if logs:
        
        for acc in test_acc:
            neptune_run["linear_eval/test_acc"].append(acc)
            
        loss = prev_run['simclr/loss'].fetch_values().value.tolist()
        for i in loss:
            neptune_run['simclr/loss'].append(i)

    prev_run.stop()




for epoch in tqdm(range(start_epoch, config['epochs']+1), desc='[Main Loop]'):
    
    random_p = 1 if epoch <= config['warmup_epochs'] else config['random_p']
    max_strength = config['max_strength']
    random_p = 0.5
    print(f'EPOCH:{epoch}    P:{random_p}  Strength:{max_strength}')
    
    
    
    if ((epoch-1) % 5 == 0):
        decoder, ppo_optimizer = ppo_init(config)
        trajectory, (img1, img2, new_img1, new_img2), entropy, (ppo_losses, ppo_rewards) = ppo_round(
            encoder=encoder, 
            decoder=decoder,
            optimizer=ppo_optimizer,
            max_strength=max_strength,
            config=config,
            neptune_run=neptune_run
        )
    
    
    
    # contrastive_round(
    #     encoder=encoder,
    #     decoder=decoder,
    #     max_strength=max_strength,
    #     epoch=epoch,
    #     config=config,
    #     optimizer=simclr_optimizer, 
    #     scheduler=simclr_scheduler, 
    #     criterion=simclr_criterion, 
    #     random_p=random_p,
    #     neptune_run=neptune_run
    # )

    if epoch % 1 == 0:
        test_acc = knn_evaluation(encoder)
    
    if logs:
        neptune_run["linear_eval/test_acc"] .append(test_acc)

    
    
    
    
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

    

if logs:
    neptune_run.stop()