import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import math
import copy
import neptune

from utils.datasets import (
    get_cifar10_dataloader, 
    rotate_images, 
    select_from_rotated_views,
)
from utils.networks import SimCLR, DecoderNN_1input, build_resnet18, build_resnet50
from utils.contrastive import InfoNCELoss, knn_evaluation, top_k_accuracy, eval_loop
from utils.ppo import (
    collect_trajectories_with_input,
    ppo_update_with_input,
    print_sorted_strings_with_counts
)
from utils.transforms import get_transforms_list, NUM_DISCREATE, transformations_dict
from utils.logs import init_neptune, get_model_save_path

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
    



def ppo_init(config: dict):

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
        lr=0.00005
    )
    
    if config['checkpoint_params']:
        checkpoint_params = config['checkpoint_params']
        decoder.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder.pt'))
        optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/decoder_opt.pt'))

    return decoder, optimizer


def contrastive_init(config: dict):
    
    if config['encoder_backbone'] == 'resnet18':
        encoder = build_resnet18()
    elif config['encoder_backbone'] == 'resnet50':
        encoder = build_resnet50()
    
    
    # encoder.load_state_dict(torch.load('params/params_523/encoder.pt'))
    encoder = encoder.to(device)

    criterion = InfoNCELoss()
    
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        momentum=0.9,
        lr=config['lr'] * config['simclr_bs'] / 256,
        weight_decay=0.0005
    )

    
    if config['checkpoint_params']:
        checkpoint_params = config['checkpoint_params']
        encoder.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder.pt'))
        optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder_opt.pt'))
    
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


def init(config: dict):
    
    encoder, simclr_optimizer, simclr_criterion = contrastive_init(config)
    decoder, ppo_optimizer = ppo_init(config)
    
    return (
        (encoder, simclr_optimizer, simclr_criterion),
        (decoder, ppo_optimizer) 
    )
    
        
def ppo_round(
        encoder: SimCLR,
        decoder: DecoderNN_1input,
        optimizer: torch.optim.Optimizer,
        config: dict,
        neptune_run: neptune.Run
    ):
    
    ppo_rounds = config['ppo_iterations']
    len_trajectory = config['ppo_len_trajectory'] 
    batch_size = config['ppo_collection_bs'] 
    ppo_epochs = config['ppo_update_epochs'] 
    ppo_batch_size = config['ppo_update_bs']
    
    if config['ppo_decoder'] == 'with_input':
        collect_trajectories = collect_trajectories_with_input
        ppo_update = ppo_update_with_input
    else:
        raise NotImplementedError
        
    
    losses = []
    rewards = []
    
    tqdm_range = tqdm(range(ppo_rounds), desc='[ppo_round]')
    for round_ in tqdm_range:
    
        trajectory, (img1, img2, new_img1, new_img2), entropy = collect_trajectories(
            len_trajectory=len_trajectory,
            encoder=encoder,
            decoder=decoder,
            batch_size=batch_size,
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
        decoder: DecoderNN_1input,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        random_p: float,
        config: dict,
        epoch: int,
        neptune_run: neptune.Run
    ):
    
    num_steps = config['simclr_iterations'] 
    batch_size = config['simclr_bs']
    
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
        
        # Simclr:
        _, z1 = encoder(x1.to(device))
        _, z2 = encoder(x2.to(device))

        sim, _, simclr_loss = criterion(z1, z2, temperature=0.5)
        
        
        # Rotation prediction:
        if config['rotation']:
            rotated_x1, rotated_labels1 = rotate_images(x1)
            rotated_x2, rotated_labels2 = rotate_images(x2)
                        
            rotated_x, rotated_labels = select_from_rotated_views(
                rotated_x1, rotated_x2,
                rotated_labels1, rotated_labels2
            )
                                
            rotated_x = rotated_x.to(device)
            rotated_labels = rotated_labels.to(device)
            
            
            feature = encoder.enc(rotated_x)
            feature = F.normalize(feature, dim=1) 
            if config['rotation_detach']:
                feature = feature.detach()
            logits = encoder.predictor(feature)
            rot_loss = F.cross_entropy(logits, rotated_labels)
            
            rot_acc = (logits.argmax(dim=-1) == rotated_labels).sum() / len(rotated_labels)
                            
        
        optimizer.zero_grad()
        simclr_loss.backward()
        if config['rotation']: 
            rot_loss.backward()
        optimizer.step()

        # logs:
        neptune_run["simclr/loss"].append(simclr_loss.item())
        neptune_run["simclr/top_5_acc"].append(top_k_accuracy(sim, 5))
        neptune_run["simclr/top_1_acc"].append(top_k_accuracy(sim, 1))
        if config['rotation']:
            neptune_run["simclr/rot_loss"].append(rot_loss.item())
            neptune_run["simclr/rot_acc"].append(rot_acc.item())


config = {
    'epochs':800,
    'warmup_epochs':10,

    'simclr_iterations':'all',
    'simclr_bs':512,
    'linear_eval_epochs':200,
    'random_p':0.5,
    'encoder_backbone': 'resnet18', # ['resnet18', 'resnet50']
    'lmbd': 0.0,
    'lr':0.03,
    'rotation':True,
    'rotation_detach':True,
    
    'ppo_decoder': 'with_input', # ['no_input', 'with_input']
    'ppo_iterations':200,
    'ppo_len_trajectory':128,
    'ppo_collection_bs':128,
    'ppo_update_bs':16,
    'ppo_update_epochs':4,
    
    'mode':'async', # ['async', 'debug']
    
    'model_save_path':model_save_path,
    'seed':seed,
    
    'checkpoint_id':"",
    'checkpoint_params':"",
    # 'checkpoint_id':"SIM-323",
    # 'checkpoint_params':'params_433',
    
}


(
    (encoder, simclr_optimizer, simclr_criterion),
    (decoder, ppo_optimizer) 
) = init(config)



logs_tags = ['random_p', 'ppo_iterations', 'model_save_path', 'rotation', 'rotation_detach']
neptune_run = init_neptune(
    tags=[f'{k}={config[k]}' for (k) in logs_tags],
    mode=config['mode']
)
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
    
    if config['mode']:
        for acc in test_acc:
            neptune_run["linear_eval/test_acc"].append(acc)
            
        loss = prev_run['simclr/loss'].fetch_values().value.tolist()
        for i in loss:
            neptune_run['simclr/loss'].append(i)

    prev_run.stop()




for epoch in tqdm(range(start_epoch, config['epochs']+1), desc='[Main Loop]'):
    
    random_p = 1 if epoch <= config['warmup_epochs'] else config['random_p']
    random_p = config['random_p']
    print(f'EPOCH:{epoch}    P:{random_p}')
    
    
    
    if (epoch > config['warmup_epochs']) and ((epoch-1) % 10 == 0):
                
        decoder, ppo_optimizer = ppo_init(config)
        trajectory, (img1, img2, new_img1, new_img2), entropy, (ppo_losses, ppo_rewards) = ppo_round(
            encoder=encoder, 
            decoder=decoder,
            optimizer=ppo_optimizer,
            config=config,
            neptune_run=neptune_run
        )
    
    
    
    contrastive_round(
        encoder=encoder,
        decoder=decoder,
        epoch=epoch,
        config=config,
        optimizer=simclr_optimizer, 
        criterion=simclr_criterion, 
        random_p=random_p,
        neptune_run=neptune_run
    )

    if epoch % 1 == 0:
        test_acc = knn_evaluation(encoder)
    
    neptune_run["linear_eval/test_acc"] .append(test_acc)

    
    
    
    
    torch.save(encoder.state_dict(), f'{model_save_path}/encoder.pt')
    torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/encoder_opt.pt')
    torch.save(decoder.state_dict(), f'{model_save_path}/decoder.pt')
    torch.save(ppo_optimizer.state_dict(), f'{model_save_path}/decoder_opt.pt')
    
    neptune_run["params/encoder"].upload(f'{model_save_path}/encoder.pt')
    neptune_run["params/encoder_opt"].upload(f'{model_save_path}/encoder_opt.pt')
    neptune_run["params/decoder"].upload(f'{model_save_path}/decoder.pt')
    neptune_run["params/decoder_opt"].upload(f'{model_save_path}/decoder_opt.pt')



print('Linear evaluation man')
accs = []
for i in range(2):
    accs.append(eval_loop(copy.deepcopy(encoder.enc), i))
line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
print(line_to_print)

    

neptune_run.stop()