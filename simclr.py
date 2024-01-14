import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, datasets
from torchvision.models import  resnet18


import neptune
from neptune.types import File



import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device:', device)


def init_neptune(tags=[]):
    run = neptune.init_run(
        project="nazim-bendib/simclr-rl",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
        tags=tags 
    )
    
    return run


class Config:
    def __init__(
        self,
        dataset='cifar10',
        data_folder='dataset',
        batch_size=256,
        num_workers=8,
        size=32
    ):
        self.dataset = dataset
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

opt = Config(
    dataset='cifar10',
    data_folder='dataset',
    batch_size=256,
    num_workers=8,
    size=32
)



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def set_loader_contrastive(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
#         normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader


# opt = Config(
#     dataset='cifar10',
#     data_folder='dataset',
#     batch_size=16,
#     num_workers=4
# )

# train_loader = set_loader_contrastive(opt)

# (x1, x2), y = next(iter(train_loader))


class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
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




class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()


    def forward(self, z1, z2, batch_size, temperature):
        features = torch.cat((z1, z2), dim=0)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        full_similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = full_similarity_matrix[~mask].view(full_similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        loss = self.CE(logits, labels)

        return full_similarity_matrix, logits, loss
    
# criterion = InfoNCELoss()


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def top_k_accuracy(sim, k):
    n_samples = sim.shape[0] // 2
    sim[range(sim.shape[0]), range(sim.shape[0])] = -1
    y_index = torch.tensor(list(range(n_samples, sim.shape[0]))).reshape(-1, 1)
    acc = (sim.argsort()[:n_samples, -k:].detach().cpu() == y_index).any(-1).sum() / n_samples
    return acc.item()



# Linear evaluation: ###########################################################################################################################



def linear_evaluation(encoder):
    
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

        
    train_dataset = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    linear_eval_model = LinearClassifier(512, num_classes=10).to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_eval_model.parameters(), lr=0.01, momentum=0.9)
    
    
    num_epochs = 10

    for epoch in tqdm(range(num_epochs), desc="Training"):
        linear_eval_model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with torch.no_grad():
                features, _ = encoder(inputs)
            outputs = linear_eval_model(features)

            # Compute loss and backpropagate
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            

    linear_eval_model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in tqdm(test_loader, desc="Evaluating test"):
            inputs, labels = inputs.to(device), labels.to(device)
            features, _ = encoder(inputs)
            outputs = linear_eval_model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = (correct / total) * 100
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(train_loader, desc="Evaluating train"):
                inputs, labels = inputs.to(device), labels.to(device)
                features, _ = encoder(inputs)
                outputs = linear_eval_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_accuracy = (correct / total) * 100
    
    return train_accuracy, test_accuracy


################################################################################################################################################







opt = Config(
    dataset='cifar10',
    data_folder='dataset',
    batch_size=512,
    num_workers=1
)

train_loader = set_loader_contrastive(opt)



model = build_resnet18().to(device)

criterion = InfoNCELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
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
        100 * len(train_loader),
        0.6,  # lr_lambda computes multiplicative factor
        1e-3
    )
)



EPOCHS = 100
PLOT_EACH = 10


logs = True
neptune_run = init_neptune(['simclr']) if logs else None



losses = []
# top_1_score = []
top_5_score = []
# top_10_score = []



for epoch in range(EPOCHS):
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, ((x1, x2), y) in tqdm_train_loader:
        
        
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        _, z1 = model(x1)
        _, z2 = model(x2)
        

        sim, _, loss = criterion(z1, z2, batch_size=x1.shape[0], temperature=0.07)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        losses.append( loss.item() )
        # top_1_score.append( top_k_accuracy(sim, 1) )
        top_5_score.append( top_k_accuracy(sim, 5) )
        # top_10_score.append( top_k_accuracy(sim, 10) )

        tqdm_train_loader.set_description(f'Epoch: {epoch+1} | Loss: {loss.item():.4f}')


        del x1, x2, loss, _
        torch.cuda.empty_cache()


        if logs:
            neptune_run["simclr/loss"].append(losses[-1])
            neptune_run["simclr/top_5_acc"].append(top_5_score[-1])
            neptune_run["simclr/sim"].append(File.as_image(np.clip(sim.cpu().detach().numpy() / 2 + 0.5, 0, 1)))

        
    
    torch.save(model.state_dict(), 'simclr_models/encoder_0.pt')
    torch.save(optimizer.state_dict(), 'simclr_models/optimizer_0.pt')
    if logs:
        neptune_run["params/model"].upload('simclr_models/encoder_0.pt')
        neptune_run["params/optimizer"].upload('simclr_models/optimizer_0.pt')
        
            
        
    
    train_acc, test_acc = linear_evaluation(model)
    if logs:
        neptune_run["simclr/linear_eval_train"].append(train_acc)
        neptune_run["simclr/linear_eval_test"].append(test_acc)
