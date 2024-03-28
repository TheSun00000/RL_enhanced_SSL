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
    def __init__(self, backbone, reduce):
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
            
        
        if reduce:
            print('reduce hein')     
            
            self.enc.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.enc.maxpool = nn.Identity()
        else:
            print('no reduce hein')     
                
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
    
    
def build_resnet18(reduce):
    return SimCLR('resnet18', reduce)

def build_resnet50(reduce):
    return SimCLR('resnet50', reduce)

  
class DecoderNN_1input(nn.Module):
    def __init__(
            self,
            transforms,
            num_discrete_magnitude,
            device,
            use_proba_head=True
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
        self.use_proba_head = use_proba_head

        self.transform_embedding = nn.Embedding(num_transforms+1, self.embed_size)
        self.magnitude_embedding = nn.Embedding(num_discrete_magnitude+1, self.embed_size)
        self.branch_id_embedding = nn.Embedding(2, self.embed_size)
        self.action_id_embedding = nn.Embedding(2, self.embed_size)

        self.rnn = nn.LSTMCell(self.embed_size * self.seq_length * 2 * 2, self.decoder_dim, bias=True)
        
        
        self.transform_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_transforms)
        )
        
        self.magnitude_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_discrete_magnitude)
        )
        
        if self.use_proba_head:
            self.proba_fc = nn.Sequential(
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
        if self.use_proba_head:
            proba_logits = self.proba_fc(h_t)
            return h_t, c_t, transform_logits, magnitude_logits, proba_logits
        return h_t, c_t, transform_logits, magnitude_logits


    def forward(self, batch_size, old_action=None):
        
        device = self.device
        
        if old_action is not None:
            old_transform_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            old_magnitude_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            if self.use_proba_head:
                old_proba_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
                
            for i in range(len(old_action)):
                for b in range(2):
                    for s in range(self.seq_length):
                        transform_id = self.transforms.index(old_action[i][b][s][0])
                        level = old_action[i][b][s][2]
                        magnitude_id = round(level * (self.num_discrete_magnitude-1))
                        old_transform_actions_index[i, b, s] = transform_id
                        old_magnitude_actions_index[i, b, s] = magnitude_id
                        if self.use_proba_head:
                            pr = old_action[i][b][s][1]
                            proba_id = round(pr * (self.num_discrete_magnitude-1))
                            old_proba_actions_index[i, b, s] = proba_id
            

        
        log_p =  torch.zeros(batch_size, 2, self.seq_length).to(device)
        
        transform_history = torch.full((batch_size, 2, self.seq_length), self.num_transforms, dtype=torch.long).to(device)
        magnitude_history = torch.full((batch_size, 2, self.seq_length), self.num_discrete_magnitude, dtype=torch.long).to(device)
        if self.use_proba_head:
            proba_history = torch.full((batch_size, 2, self.seq_length), self.num_discrete_magnitude, dtype=torch.long).to(device)
            

        transform_entropy = 0
        magnitude_entropy = 0
        proba_entropy = 0
        
        # Initialize LSTM state
        h_t, c_t = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        for branch in range(2):
            
            for step in range(self.seq_length):
                
                if not self.use_proba_head: 
                    
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
                    
                else: # if self.use_proba_head:
                    
                    h_t, c_t, transform_logits, magnitude_logits, proba_logits = self.lstm_forward(
                        transform_history=transform_history,
                        magnitude_history=magnitude_history,
                        h_t=h_t,
                        c_t=c_t,
                    )
                    
                    if old_action is None:
                        transform_action_index = Categorical(logits=transform_logits).sample()
                        magnitude_action_index = Categorical(logits=magnitude_logits).sample()
                        proba_action_index = Categorical(logits=proba_logits).sample()
                    else:
                        transform_action_index = old_transform_actions_index[:, branch, step]
                        magnitude_action_index = old_magnitude_actions_index[:, branch, step]
                        proba_action_index = old_proba_actions_index[:, branch, step]
                                    
                    
                    transform_log_p = F.log_softmax(transform_logits, dim=-1).gather(-1,transform_action_index.unsqueeze(-1))
                    magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,magnitude_action_index.unsqueeze(-1))
                    proba_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,proba_action_index.unsqueeze(-1))
                    
                    log_p[:, branch, step] = transform_log_p.squeeze(-1) + magnitude_log_p.squeeze(-1) + proba_log_p.squeeze(-1)
                    
                    transform_entropy += Categorical(logits=transform_logits).entropy().mean()
                    magnitude_entropy += Categorical(logits=magnitude_logits).entropy().mean()
                    proba_entropy += Categorical(logits=proba_logits).entropy().mean()
                    
                    transform_history = transform_history.clone() 
                    transform_history[range(batch_size), branch, step] = transform_action_index
                    magnitude_history = magnitude_history.clone() 
                    magnitude_history[range(batch_size), branch, step] = magnitude_action_index
                    proba_history = proba_history.clone() 
                    proba_history[range(batch_size), branch, step] = proba_action_index
                



        transform_entropy /= (2*self.seq_length)
        magnitude_entropy /= (2*self.seq_length)
        entropy = transform_entropy + magnitude_entropy
        
        if self.use_proba_head:
            entropy += proba_entropy / (2*self.seq_length)
        
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
                    
                    pr = 0.8
                    if self.use_proba_head:
                        pr = (proba_history[i, b, s] / (self.num_discrete_magnitude-1)).item()    
                    
                    action[-1][b].append((
                        self.transforms[transform_history[i, b, s]],
                        pr,
                        level
                    ))
        
        return (
            log_p,
            action,
            entropy
        )
    
    def get_policy_list(self, N=1000):
        _, policy, _ = self.forward(N)
        return policy
    
    
    
    

  
class DecoderNN_1input_one_branch(nn.Module):
    def __init__(
            self,
            transforms,
            num_discrete_magnitude,
            device,
            use_proba_head=True
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
        self.use_proba_head = use_proba_head

        self.transform_embedding = nn.Embedding(num_transforms+1, self.embed_size)
        self.magnitude_embedding = nn.Embedding(num_discrete_magnitude+1, self.embed_size)
        self.branch_id_embedding = nn.Embedding(2, self.embed_size)
        self.action_id_embedding = nn.Embedding(2, self.embed_size)

        self.rnn = nn.LSTMCell(self.embed_size * self.seq_length * 2, self.decoder_dim, bias=True)
        
        
        self.transform_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_transforms)
        )
        
        self.magnitude_fc = nn.Sequential(
            nn.Linear(self.decoder_dim,512),
            nn.ReLU(),
            nn.Linear(512,num_discrete_magnitude)
        )
        
        if self.use_proba_head:
            self.proba_fc = nn.Sequential(
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
        if self.use_proba_head:
            proba_logits = self.proba_fc(h_t)
            return h_t, c_t, transform_logits, magnitude_logits, proba_logits
        return h_t, c_t, transform_logits, magnitude_logits


    def forward(self, batch_size, old_action=None):
        
        device = self.device
        
        if old_action is not None:
            old_transform_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            old_magnitude_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
            if self.use_proba_head:
                old_proba_actions_index = torch.zeros((batch_size, 2, self.seq_length), dtype=torch.long).to(device)
                
            for i in range(len(old_action)):
                for b in range(2):
                    for s in range(self.seq_length):
                        transform_id = self.transforms.index(old_action[i][b][s][0])
                        level = old_action[i][b][s][2]
                        magnitude_id = round(level * (self.num_discrete_magnitude-1))
                        old_transform_actions_index[i, b, s] = transform_id
                        old_magnitude_actions_index[i, b, s] = magnitude_id
                        if self.use_proba_head:
                            pr = old_action[i][b][s][1]
                            proba_id = round(pr * (self.num_discrete_magnitude-1))
                            old_proba_actions_index[i, b, s] = proba_id
            

        transform_history = torch.full((batch_size, 2, self.seq_length), self.num_transforms, dtype=torch.long).to(device)
        magnitude_history = torch.full((batch_size, 2, self.seq_length), self.num_discrete_magnitude, dtype=torch.long).to(device)
        if self.use_proba_head:
            proba_history = torch.full((batch_size, 2, self.seq_length), self.num_discrete_magnitude, dtype=torch.long).to(device)
    
        log_p =  torch.zeros(batch_size, 2, self.seq_length).to(device)

        transform_entropy = 0
        magnitude_entropy = 0
        proba_entropy = 0
    
        
        for branch in range(2):
            
            # Initialize LSTM state
            h_t, c_t = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
            
            for step in range(self.seq_length):
                
                if not self.use_proba_head: 
                    
                    h_t, c_t, transform_logits, magnitude_logits = self.lstm_forward(
                        transform_history=transform_history[:, branch, :],
                        magnitude_history=magnitude_history[:, branch, :],
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
                    
                else: # if self.use_proba_head:
                    
                    h_t, c_t, transform_logits, magnitude_logits, proba_logits = self.lstm_forward(
                        transform_history=transform_history[:, branch, :],
                        magnitude_history=magnitude_history[:, branch, :],
                        h_t=h_t,
                        c_t=c_t,
                    )
                    
                    if old_action is None:
                        transform_action_index = Categorical(logits=transform_logits).sample()
                        magnitude_action_index = Categorical(logits=magnitude_logits).sample()
                        proba_action_index = Categorical(logits=proba_logits).sample()
                    else:
                        transform_action_index = old_transform_actions_index[:, branch, step]
                        magnitude_action_index = old_magnitude_actions_index[:, branch, step]
                        proba_action_index = old_proba_actions_index[:, branch, step]
                                    
                    
                    transform_log_p = F.log_softmax(transform_logits, dim=-1).gather(-1,transform_action_index.unsqueeze(-1))
                    magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,magnitude_action_index.unsqueeze(-1))
                    proba_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,proba_action_index.unsqueeze(-1))
                    
                    log_p[:, branch, step] = transform_log_p.squeeze(-1) + magnitude_log_p.squeeze(-1) + proba_log_p.squeeze(-1)
                    
                    transform_entropy += Categorical(logits=transform_logits).entropy().mean()
                    magnitude_entropy += Categorical(logits=magnitude_logits).entropy().mean()
                    proba_entropy += Categorical(logits=proba_logits).entropy().mean()
                    
                    transform_history = transform_history.clone() 
                    transform_history[range(batch_size), branch, step] = transform_action_index
                    magnitude_history = magnitude_history.clone() 
                    magnitude_history[range(batch_size), branch, step] = magnitude_action_index
                    proba_history = proba_history.clone() 
                    proba_history[range(batch_size), branch, step] = proba_action_index
                



        transform_entropy /= (2*self.seq_length)
        magnitude_entropy /= (2*self.seq_length)
        entropy = transform_entropy + magnitude_entropy
        
        if self.use_proba_head:
            entropy += proba_entropy / (2*self.seq_length)
        
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
                    
                    pr = 0.8
                    if self.use_proba_head:
                        pr = (proba_history[i, b, s] / (self.num_discrete_magnitude-1)).item()    
                    
                    action[-1][b].append((
                        self.transforms[transform_history[i, b, s]],
                        pr,
                        level
                    ))
        
        return (
            log_p,
            action,
            entropy
        )
    
    def get_policy_list(self, N=1000):
        _, policy, _ = self.forward(N)
        return policy
    
    