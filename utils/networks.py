import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision.models import resnet18
from itertools import permutations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device





class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLR, self).__init__()
        self.enc = resnet18(weights=None)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
    
def build_resnet18():
    return SimCLR()




class DecoderRNN(nn.Module):
    def __init__(
            self,
            embed_size, 
            encoder_dim, 
            decoder_dim,
            num_transforms=4,
            num_discrete_magnitude=11,
            seq_length=10,
            drop_prob=0.3
        ):
        super().__init__()
        
        #save the model param
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.num_transforms = num_transforms
        self.num_discrete_magnitude = num_discrete_magnitude
        self.seq_length = seq_length


        len_action = num_transforms + num_discrete_magnitude + 1 # +1 for the first operation
        self.action_embedding = nn.Embedding(len_action, embed_size)
        self.branch_id_embedding = nn.Embedding(2, embed_size)
        self.action_id_embedding = nn.Embedding(2, embed_size)

        # z + branch_id_embd + prev_action_embd
        self.rnn = nn.LSTMCell(encoder_dim + embed_size + embed_size + embed_size, decoder_dim, bias=True)        
        
        self.fcn_transform = nn.Linear(decoder_dim,num_transforms)
        self.fcn_magnitude = nn.Linear(decoder_dim,num_discrete_magnitude)

    

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.decoder_dim, device=device)
        c = torch.zeros(batch_size, self.decoder_dim, device=device)
        return h, c
    

    def lstm_forward(self, z, branch_id, action_id, prev_action, action_mask, h_t, c_t, decoder):
        """
        z: the representation of the image
        branch_id: 0 or 1, to know which branch are we in (z1 or z2)
        action_id: 0 or 1, to know whather we are selecting a transformation or a magnitude
        prev_action: the index of the previous action taken (transformation and magnitude)
        action_mask: transformation action mask
        h_t: hidden state of the lstm
        c_t: context of the lstmd
        decoder: the decoder (the classifier)
        """
        branch_id_embd   = self.branch_id_embedding(branch_id)
        action_id_embd   = self.action_id_embedding(action_id)
        prev_action_embd = self.action_embedding(prev_action)
        input = torch.concat(
            (z, branch_id_embd, action_id_embd, prev_action_embd),
            dim=-1
        )
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        # print('before:', logits[0])
        if action_mask is not None:
            inf_tensor = torch.full_like(logits, -float('inf'))
            logits = torch.where(action_mask, logits, inf_tensor)
        # print('after :', logits[0])
        # print('-----')
        return h_t, c_t, logits


    def forward(self, z1, z2, old_action_index=None):
        """
        returns:
            transform_actions_index: (batch_size, 2, seq_length)
            magnitude_actions_index: (batch_size, 2, seq_length)
            transform_log_p: (batch_size, 2, seq_length)
            magnitude_log_p: (batch_size, 2, seq_length)
        """
        if old_action_index is not None:
            (old_transform_actions_index, old_magnitude_actions_index) = old_action_index

        #get the seq length to iterate
        seq_length = self.seq_length
        batch_size = z1.size(0)
        
        transform_log_p =  torch.zeros(batch_size, 2, seq_length).to(device)
        transform_actions_index =  torch.zeros(batch_size, 2, seq_length, dtype=torch.long).to(device)

        magnitude_log_p =  torch.zeros(batch_size, 2, seq_length).to(device)
        magnitude_actions_index =  torch.zeros(batch_size, 2, seq_length, dtype=torch.long).to(device)

        branch_id = torch.full((batch_size,2), 0, dtype=torch.long, device=device)
        branch_id[:, 1] = 1
        
        action_id = torch.full((batch_size,2), 0, dtype=torch.long, device=device)
        action_id[:, 1] = 1
        
        features = [z1, z2]
        action_index = torch.LongTensor([self.num_transforms + self.num_discrete_magnitude]).to(device)
        action_index = action_index.repeat(batch_size)

        transform_entropy = 0
        magnitude_entropy = 0
        
        # Initialize LSTM state
        h_t, c_t = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        for branch in range(2):

            z = features[branch]
            action_mask = torch.ones((batch_size, self.num_transforms), dtype=torch.bool, requires_grad=False).to(device)

            for step in range(seq_length):
                
                
                
                h_t, c_t, transform_logits = self.lstm_forward(
                    z=z,
                    branch_id=branch_id[:, branch],
                    action_id=action_id[:, 0],
                    prev_action=action_index,
                    action_mask=action_mask,
                    h_t=h_t,
                    c_t=c_t,
                    decoder=self.fcn_transform
                )
                if old_action_index is None:
                    action_index = Categorical(logits=transform_logits).sample()
                else:
                    action_index = old_transform_actions_index[:, branch, step]
                                
                action_mask = action_mask.clone()
                action_mask[range(batch_size), action_index] = 0
                
                log_p = F.log_softmax(transform_logits, dim=-1).gather(-1,action_index.unsqueeze(-1))
                transform_log_p[:, branch, step] = log_p.squeeze(-1)
                transform_actions_index[:, branch, step] = action_index
                transform_entropy += Categorical(logits=transform_logits).entropy().mean()



                h_t, c_t, magnitude_logits = self.lstm_forward(
                    z=z,
                    branch_id=branch_id[:, branch],
                    action_id=action_id[:, 1],
                    prev_action=action_index,
                    action_mask=None,
                    h_t=h_t,
                    c_t=c_t,
                    decoder=self.fcn_magnitude
                )
                if old_action_index is None:
                    action_index = Categorical(logits=magnitude_logits).sample()
                else:
                    action_index = old_magnitude_actions_index[:, branch, step]
                
                log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1,action_index.unsqueeze(-1))
                magnitude_log_p[:, branch, step] = log_p.squeeze(-1)
                magnitude_actions_index[:, branch, step] = action_index
                magnitude_entropy += Categorical(logits=magnitude_logits).entropy().mean()


        transform_entropy /= (2*seq_length)
        magnitude_entropy /= (2*seq_length)
        
        log_p = transform_log_p.reshape(batch_size, -1).sum(-1) + magnitude_log_p.reshape(batch_size, -1).sum(-1)
        log_p = log_p.unsqueeze(-1)

        # log_p.shape == (batch_size, 1)
        # transform_actions_index.shape == (batch_size, 2, 4)
        # magnitude_actions_index.shape == (batch_size, 2, 4)
        # transform_entropy.shape == ()
        # magnitude_entropy.shape == ()
        
        return (
                log_p,
                (transform_actions_index, magnitude_actions_index),
                (transform_entropy, magnitude_entropy)
            )
        
        
        

class DecoderNoInput(nn.Module):
    def __init__(self,
            num_transforms,
            num_discrete_magnitude,
            device
            ):
        super().__init__()
    
    #save the model param

        self.num_transforms = num_transforms
        self.num_discrete_magnitude = num_discrete_magnitude
        self.seq_length = num_transforms
        self.device = device
        
        self.permutations = torch.tensor(
            list(permutations(range(4)))
            ).to(device)
        
        self.num_transforms_permutations = len(self.permutations)
        self.num_actions = num_transforms * num_discrete_magnitude
        
        self.model = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.num_actions + 2 * self.num_transforms_permutations),
        )
        
        
    def forward(self, batch_size, old_action_index=None):
        
        x = torch.zeros((batch_size,1), dtype=torch.float32).to(self.device)
        
        output = self.model(x)
        
        magnitude_logits = output[:, :2 * self.num_actions]
        permutations_logits = output[:, 2 * self.num_actions:]
        
        magnitude_logits = magnitude_logits.reshape(batch_size, 2, self.num_transforms, self.num_discrete_magnitude)
        permutations_logits = permutations_logits.reshape(batch_size, 2, self.num_transforms_permutations)
        
        magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)
        permutations_dist = torch.distributions.Categorical(logits=permutations_logits)
        
        if old_action_index is None:
            magnitude_actions_index = magnitude_dist.sample()
            permutations_index = permutations_dist.sample()
        else:
            transform_actions_index, magnitude_actions_index = old_action_index
            matches = torch.all(transform_actions_index.unsqueeze(0) == self.permutations.unsqueeze(1).unsqueeze(1), dim=-1) * 1
            permutations_index = torch.argmax(matches, dim=0)
            magnitude_actions_index = magnitude_actions_index
                
        magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1, magnitude_actions_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        permutation_log_p = F.log_softmax(permutations_logits, dim=-1).gather(-1, permutations_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        
        log_p = magnitude_log_p + permutation_log_p
        transform_actions_index = self.permutations[permutations_index]
        magnitude_actions_index = magnitude_actions_index
        transform_entropy = permutations_dist.entropy().mean()
        magnitude_entropy = magnitude_dist.entropy().mean()
        
        # print(log_p.shape)
        # print(transform_actions_index.shape)
        # print(magnitude_actions_index.shape)
        # print(transform_entropy.shape)
        # print(magnitude_entropy.shape)
        
        return (
                log_p,
                (transform_actions_index, magnitude_actions_index),
                (transform_entropy, magnitude_entropy)
            )        