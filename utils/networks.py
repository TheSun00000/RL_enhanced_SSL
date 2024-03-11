import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.resnet import resnet18, resnet50
from itertools import permutations
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device




def count_occurrences(list_of_lists):
    counts = Counter(tuple(sublist) for sublist in list_of_lists)
    counts = [(list(sublist), count) for sublist, count in counts.items()]
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    return counts


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super(SimCLR, self).__init__()
        
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.enc = resnet18()
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.enc = resnet50()
            self.feature_dim = 2048
        else:
            raise NotImplementedError  
            
        
                
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, affine=False)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4)
        )
        
        # self.predictor = nn.Sequential(
        #     nn.Linear(self.feature_dim, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 4)
        # )

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
    
def build_resnet18():
    return SimCLR('resnet18')

def build_resnet50():
    return SimCLR('resnet50')


class DecoderNN_1input_(nn.Module):
    def __init__(self,
            num_transforms,
            num_discrete_magnitude,
            device
            ):
        super().__init__()
    
        self.num_transforms = num_transforms
        self.num_discrete_magnitude = num_discrete_magnitude
        self.seq_length = num_transforms
        self.device = device
        
        self.permutations = torch.tensor(
            list(permutations(range(4)))
            ).to(device)
        
        self.num_transforms_permutations = len(self.permutations)
        self.num_actions = num_transforms * num_discrete_magnitude
        
        self.output_dim_per_view = (
            # Crop (Position[10], Area[10]):
            num_discrete_magnitude + num_discrete_magnitude + \
            # Color (4*magnitude[10], permutations):
            4 * num_discrete_magnitude + self.num_transforms_permutations + \
            # Gray (Porba[10]):
            num_discrete_magnitude + \
            # Gaussian blur (Sigma[10], Proba[10]):
            num_discrete_magnitude + num_discrete_magnitude
        )
        
        self.model = nn.Sequential(
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(2048, 2 * self.output_dim_per_view)
        )
        
        
    def forward(self, x=None, batch_size=None, old_action_index=None):
        
        if x is None:
            assert batch_size is not None, "batch_size should be specified"
            x = torch.zeros((batch_size, 2048), dtype=torch.float32).to(device)
            
        *leading_dim, input_dim = x.shape
        
        x = F.normalize(x, dim=-1)                     
        output = self.model(x)
        
        D = self.num_discrete_magnitude
        crop_position_offset = 0
        crop_area_offset = crop_position_offset + 2*D
        color_magnitude_offset = crop_area_offset + 2*D
        color_permutation_offset = color_magnitude_offset + 2*(4*D)
        gray_proba_offset = color_permutation_offset + 2*self.num_transforms_permutations
        blur_sigma_offset = gray_proba_offset + 2*D
        blur_proba_offset = blur_sigma_offset + 2*D
                
        
        crop_position_logits = output[:, :crop_area_offset]
        crop_area_logits = output[:, crop_area_offset:color_magnitude_offset]
        color_magnitude_logits = output[:, color_magnitude_offset:color_permutation_offset]
        color_permutation_logits = output[:, color_permutation_offset:gray_proba_offset]
        gray_proba_logits = output[:, gray_proba_offset:blur_sigma_offset]
        blur_sigma_logits = output[:, blur_sigma_offset:blur_proba_offset]
        blur_proba_logits = output[:, blur_proba_offset:]

        
        crop_position_logits = crop_position_logits.reshape(*leading_dim, 2, D)
        crop_area_logits = crop_area_logits.reshape(*leading_dim, 2, D)
        color_magnitude_logits = color_magnitude_logits.reshape(*leading_dim, 2, 4, D)
        color_permutation_logits = color_permutation_logits.reshape(*leading_dim, 2, self.num_transforms_permutations)
        gray_proba_logits = gray_proba_logits.reshape(*leading_dim, 2, D)
        blur_sigma_logits = blur_sigma_logits.reshape(*leading_dim, 2, D)
        blur_proba_logits = blur_proba_logits.reshape(*leading_dim, 2, D)
        
        crop_position_dist = torch.distributions.Categorical(logits=crop_position_logits)
        crop_area_dist = torch.distributions.Categorical(logits=crop_area_logits)
        color_magnitude_dist = torch.distributions.Categorical(logits=color_magnitude_logits)
        color_permutation_dist = torch.distributions.Categorical(logits=color_permutation_logits)
        gray_proba_dist = torch.distributions.Categorical(logits=gray_proba_logits)
        blur_sigma_dist = torch.distributions.Categorical(logits=blur_sigma_logits)
        blur_proba_dist = torch.distributions.Categorical(logits=blur_proba_logits)
        
        # print('crop_position:', crop_position_dist.entropy().mean())
        # print('crop_area:', crop_area_dist.entropy().mean())
        # print('color_magnitude:', color_magnitude_dist.entropy().mean())
        # print('color_permutation:', color_permutation_dist.entropy().mean())
        # print('gray_proba:', gray_proba_dist.entropy().mean())
        # print('blur_sigma:', blur_sigma_dist.entropy().mean())
        # print('blur_proba:', blur_proba_dist.entropy().mean())
        # print('------------------')
                
        
        if old_action_index is None:
            crop_position_index = crop_position_dist.sample().clip(0, 8)
            crop_area_index = crop_area_dist.sample()
            color_magnitude_index = color_magnitude_dist.sample()
            color_permutation_index = color_permutation_dist.sample()
            gray_proba_index = gray_proba_dist.sample()
            blur_sigma_index = blur_sigma_dist.sample()
            blur_proba_index = blur_proba_dist.sample()
        else:
            crop_position_index = old_action_index[..., 0]
            crop_area_index = old_action_index[..., 1]
            color_magnitude_index = old_action_index[..., 2:6]
            color_permutation_index = old_action_index[..., 6]
            gray_proba_index = old_action_index[..., 7]
            blur_sigma_index = old_action_index[..., 8]
            blur_proba_index = old_action_index[..., 9]
        
        
        crop_position_log_p = F.log_softmax(crop_position_logits, dim=-1).gather(-1, crop_position_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        crop_area_log_p = F.log_softmax(crop_area_logits, dim=-1).gather(-1, crop_area_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        color_magnitude_log_p = F.log_softmax(color_magnitude_logits, dim=-1).gather(-1, color_magnitude_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        color_permutation_log_p = F.log_softmax(color_permutation_logits, dim=-1).gather(-1, color_permutation_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        gray_proba_log_p = F.log_softmax(gray_proba_logits, dim=-1).gather(-1, gray_proba_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        blur_sigma_log_p = F.log_softmax(blur_sigma_logits, dim=-1).gather(-1, blur_sigma_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)
        blur_proba_log_p = F.log_softmax(blur_proba_logits, dim=-1).gather(-1, blur_proba_index.unsqueeze(-1)).reshape(*leading_dim, -1).sum(-1, keepdim=True)


        # log_p = (crop_position_log_p + crop_area_log_p) + (color_magnitude_log_p + color_permutation_log_p) + (gray_proba_log_p) + (blur_sigma_log_p + blur_proba_log_p)
        log_p = (color_magnitude_log_p + color_permutation_log_p)
        # log_p = (color_magnitude_log_p + color_permutation_log_p)
        
        actions_index = torch.concat((
            crop_position_index.unsqueeze(-1),
            crop_area_index.unsqueeze(-1),
            color_magnitude_index,
            color_permutation_index.unsqueeze(-1),
            gray_proba_index.unsqueeze(-1),
            blur_sigma_index.unsqueeze(-1),
            blur_proba_index.unsqueeze(-1),
        ), dim=-1)
                
        # entropy = (crop_position_dist.entropy().mean() + crop_area_dist.entropy().mean()) + (color_magnitude_dist.entropy().mean() + color_permutation_dist.entropy().mean()) + (gray_proba_dist.entropy().mean()) + (blur_sigma_dist.entropy().mean() + blur_proba_dist.entropy().mean())
        entropy = (color_magnitude_dist.entropy().mean() + color_permutation_dist.entropy().mean())
        
        return (
                log_p,
                actions_index,
                entropy
            )
        
    
    def sample(self, num_samples):
        x = torch.zeros((1, 2048), dtype=torch.float32).to(self.device)
        
        *leading_dim, input_dim = x.shape
        
        x = F.normalize(x, dim=-1)                     
        output = self.model(x)
        
        D = self.num_discrete_magnitude
        crop_position_offset = 0
        crop_area_offset = crop_position_offset + 2*D
        color_magnitude_offset = crop_area_offset + 2*D
        color_permutation_offset = color_magnitude_offset + 2*(4*D)
        gray_proba_offset = color_permutation_offset + 2*self.num_transforms_permutations
        blur_sigma_offset = gray_proba_offset + 2*D
        blur_proba_offset = blur_sigma_offset + 2*D
                
        color_magnitude_logits = output[:, color_magnitude_offset:color_permutation_offset]
        color_permutation_logits = output[:, color_permutation_offset:gray_proba_offset]

        color_magnitude_logits = color_magnitude_logits.reshape(*leading_dim, 2, 4, D)
        color_permutation_logits = color_permutation_logits.reshape(*leading_dim, 2, self.num_transforms_permutations)
        
        color_magnitude_dist = torch.distributions.Categorical(logits=color_magnitude_logits)
        color_permutation_dist = torch.distributions.Categorical(logits=color_permutation_logits)
        
        color_magnitude_index, color_permutation_index = color_magnitude_dist.sample((num_samples,)), color_permutation_dist.sample((num_samples,))

        actions_index = torch.concat((
            torch.zeros((num_samples, 2, 1), dtype=torch.int64).to(self.device),
            torch.zeros((num_samples, 2, 1), dtype=torch.int64).to(self.device),
            color_magnitude_index.squeeze(1),
            color_permutation_index.squeeze(1).unsqueeze(-1),
            torch.zeros((num_samples, 2, 1), dtype=torch.int64).to(self.device),
            torch.zeros((num_samples, 2, 1), dtype=torch.int64).to(self.device),
            torch.zeros((num_samples, 2, 1), dtype=torch.int64).to(self.device),
        ), dim=-1)

        
        return actions_index
    
    
class DecoderNN_1input(nn.Module):
    def __init__(
            self,
            transforms,
            num_discrete_magnitude,
            device
        ):
        super().__init__()
        
        #save the model param
        self.encoder_dim = 2028
        self.decoder_dim = 512
        self.embed_size = 128

        self.transforms = transforms
        num_transforms = len(transforms)
        self.num_transforms = num_transforms
        self.num_discrete_magnitude = num_discrete_magnitude
        self.seq_length = 2

        self.transform_embedding = nn.Embedding(num_transforms+1, self.embed_size)
        self.magnitude_embedding = nn.Embedding(num_discrete_magnitude+1, self.embed_size)
        self.branch_id_embedding = nn.Embedding(2, self.embed_size)
        self.action_id_embedding = nn.Embedding(2, self.embed_size)

        self.rnn = nn.LSTMCell(self.embed_size * self.seq_length * 2 * 2, self.decoder_dim, bias=True)
        
        # self.transform_fc = nn.Linear(self.decoder_dim,num_transforms)
        self.transform_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_transforms)
        )
        # self.magnitude_fc = nn.Linear(self.decoder_dim,num_discrete_magnitude)
        self.magnitude_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_discrete_magnitude)
        )
        
        self.device = device

    

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.decoder_dim, device=device)
        c = torch.zeros(batch_size, self.decoder_dim, device=device)
        return h, c
    

    def lstm_forward(self, transform_history, magnitude_history, h_t, c_t):
        
        batch_size = transform_history.shape[0]
        
        transform_history_embd = self.transform_embedding(transform_history)
        magnitude_history_embd = self.magnitude_embedding(magnitude_history)
        input = torch.concat(
            (transform_history_embd, magnitude_history_embd),
            dim=-1
        ).reshape(batch_size, -1)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        transform_logits = self.transform_fc(h_t)
        magnitude_logits = self.magnitude_fc(h_t)
        return h_t, c_t, transform_logits, magnitude_logits


    def forward(self, batch_size, old_action=None):
        
        device = self.device
        
        if old_action is not None:
            old_transform_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            old_magnitude_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            for i in range(len(old_action)):
                for b in range(2):
                    for s in range(self.seq_length):
                        transform_id = self.transforms.index(old_action[i][b][s][0])
                        level = old_action[i][b][s][2]
                        magnitude_id = round(level * (self.num_discrete_magnitude-1))
                        old_transform_actions_index[i, b, s] = transform_id
                        old_magnitude_actions_index[i, b, s] = magnitude_id
            

        
        log_p =  torch.zeros(batch_size, 2, self.seq_length).to(device)
        
        transform_history = torch.full((batch_size, 2, self.seq_length), self.num_transforms, dtype=torch.long).to(device)
        magnitude_history = torch.full((batch_size, 2, self.seq_length), self.num_discrete_magnitude, dtype=torch.long).to(device)

        transform_entropy = 0
        magnitude_entropy = 0
        
        # Initialize LSTM state
        h_t, c_t = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        for branch in range(2):
            
            for step in range(self.seq_length):

                h_t, c_t, transform_logits, magnitude_logits = self.lstm_forward(
                    transform_history=transform_history,
                    magnitude_history=magnitude_history,
                    h_t=h_t,
                    c_t=c_t,
                )
                
                
                if old_action is None:
                    transform_action_index = Categorical(logits=transform_logits).sample()
                    magnitude_action_index = Categorical(logits=magnitude_logits).sample()
                else:
                    transform_action_index = old_transform_actions_index[:, branch, step]
                    magnitude_action_index = old_magnitude_actions_index[:, branch, step]
                                
                
                transform_log_p = F.log_softmax(transform_logits, dim=-1).gather(-1,transform_action_index.unsqueeze(-1))
                magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,magnitude_action_index.unsqueeze(-1))
                
                log_p[:, branch, step] = transform_log_p.squeeze(-1) + magnitude_log_p.squeeze(-1)
                
                transform_entropy += Categorical(logits=transform_logits).entropy().mean()
                magnitude_entropy += Categorical(logits=magnitude_logits).entropy().mean()
                
                transform_history = transform_history.clone() 
                transform_history[range(batch_size), branch, step] = transform_action_index
                magnitude_history = magnitude_history.clone() 
                magnitude_history[range(batch_size), branch, step] = magnitude_action_index
                



        transform_entropy /= (2*self.seq_length)
        magnitude_entropy /= (2*self.seq_length)
        entropy = transform_entropy + magnitude_entropy
        
        log_p = log_p.reshape(batch_size, -1).sum(-1) 
        log_p = log_p.unsqueeze(-1)
        
        action = []

        for i in range(batch_size):
            action.append([])
            action[-1].append([])
            action[-1].append([])
            for b in range(2):
                for s in range(self.seq_length):
                    level = (magnitude_history[i, b, s] / (self.num_discrete_magnitude-1)).item()                    
                    action[-1][b].append((
                        self.transforms[transform_history[i, b, s]],
                        0.8,
                        level
                    ))
        
        return (
            log_p,
            action,
            entropy
        )
    
    def get_policy_list(self, N=20000):
        _, policy, _ = self.forward(N)
        return policy