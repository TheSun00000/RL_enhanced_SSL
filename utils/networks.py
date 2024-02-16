import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision.models import resnet18, resnet50
from itertools import permutations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device





class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super(SimCLR, self).__init__()
        if backbone == 'resnet18':
            self.enc = resnet18(weights=None)
        elif backbone == 'resnet50':
            self.enc = resnet50(weights=None)
        else:
            raise     
            
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        # self.projector = nn.Sequential(
        #     nn.Linear(self.feature_dim, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, projection_dim)
        # )
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
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

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
    
def build_resnet18():
    return SimCLR('resnet18')

def build_resnet50():
    return SimCLR('resnet50')




class DecoderLSTM(nn.Module):
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
        

class DecoderNN_2inputs(nn.Module):
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
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * self.num_actions + 2 * self.num_transforms_permutations),
        )
        
        
    def forward(self, x, old_action_index=None):
                
        batch_size = x.shape[0]
                
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
            matches = torch.all(transform_actions_index[...,:4].unsqueeze(0) == self.permutations.unsqueeze(1).unsqueeze(1), dim=-1) * 1
            permutations_index = torch.argmax(matches, dim=0)
            magnitude_actions_index = magnitude_actions_index
                
        magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1, magnitude_actions_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        permutation_log_p = F.log_softmax(permutations_logits, dim=-1).gather(-1, permutations_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        
        log_p = magnitude_log_p + permutation_log_p

        transform_actions_index = self.permutations[permutations_index]
        grayscale_tensor = torch.full_like(transform_actions_index[..., :1], transform_actions_index.max()+1)
        transform_actions_index = torch.concat((transform_actions_index, grayscale_tensor), dim=-1)
        
        magnitude_actions_index = magnitude_actions_index
        transform_entropy = permutations_dist.entropy().mean()
        magnitude_entropy = magnitude_dist.entropy().mean()
                
        print(log_p.shape)
        print(transform_actions_index.shape)
        print(magnitude_actions_index.shape)
        print(transform_entropy.shape)
        print(magnitude_entropy.shape)
        
        
        # torch.Size([64, 1])
        # torch.Size([64, 2, 5])
        # torch.Size([64, 2, 5])
        # torch.Size([])
        # torch.Size([])

        
        return (
                log_p,
                (transform_actions_index, magnitude_actions_index),
                (transform_entropy, magnitude_entropy)
            )


class DecoderNN_1input(nn.Module):
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
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * self.output_dim_per_view)
        )
        
        
    def forward(self, x, old_action_index=None):
        
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
        log_p = (crop_position_log_p + crop_area_log_p) + (color_magnitude_log_p + color_permutation_log_p) + (gray_proba_log_p)
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
        entropy = (crop_position_dist.entropy().mean() + crop_area_dist.entropy().mean()) + (color_magnitude_dist.entropy().mean() + color_permutation_dist.entropy().mean()) + (gray_proba_dist.entropy().mean())
        
        return (
                log_p,
                actions_index,
                entropy
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
            matches = torch.all(transform_actions_index[...,:4].unsqueeze(0) == self.permutations.unsqueeze(1).unsqueeze(1), dim=-1) * 1
            permutations_index = torch.argmax(matches, dim=0)
            magnitude_actions_index = magnitude_actions_index
                
        magnitude_log_p = F.log_softmax(magnitude_logits, dim=-1).gather(-1, magnitude_actions_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        permutation_log_p = F.log_softmax(permutations_logits, dim=-1).gather(-1, permutations_index.unsqueeze(-1)).reshape(batch_size, -1).sum(-1, keepdim=True)
        
        log_p = magnitude_log_p + permutation_log_p
        
        transform_actions_index = self.permutations[permutations_index]
        grayscale_tensor = torch.full_like(transform_actions_index[..., :1], transform_actions_index.max()+1)
        transform_actions_index = torch.concat((transform_actions_index, grayscale_tensor), dim=-1)
        
        magnitude_actions_index = magnitude_actions_index
        
        transform_entropy = permutations_dist.entropy().mean()
        magnitude_entropy = magnitude_dist.entropy().mean()
        
        
        return (
                log_p,
                (transform_actions_index, magnitude_actions_index),
                (transform_entropy, magnitude_entropy)
            )        


