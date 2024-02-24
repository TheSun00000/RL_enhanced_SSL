import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm

from utils.datasets2 import get_essl_memory_loader, get_essl_test_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask



class InfoNCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(InfoNCELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction=reduction)


    def forward(self, z1, z2, temperature):
        
        batch_size = z1.shape[0]
        
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
    


class InfoNCELoss_(nn.Module):
    def __init__(self, reduction='mean'):
        super(InfoNCELoss_, self).__init__()
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(reduction=reduction)


    def forward(self, z1, z2, temperature):
        
        batch_size = z1.shape[0]
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # neg score
        out = torch.cat([z1, z2], dim=0)
        sim = torch.mm(out, out.t().contiguous())
        neg = torch.exp( sim / temperature)
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        Ng = neg.sum(dim=-1)
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) ))
        if self.reduction == 'mean':
            loss = loss.mean()
        
        # print(sim.shape, loss)

        return sim, None, loss
    

def info_nce_loss(z1, z2, temperature=0.5, reduction='mean'):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)
    return loss

class InfoNCELoss_(nn.Module):
    def __init__(self, reduction='mean'):
        super(InfoNCELoss_, self).__init__()
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(self, z1, z2, temperature):
        loss = info_nce_loss(z1, z2, temperature, self.reduction) / 2 + info_nce_loss(z2, z1, temperature, self.reduction) / 2
        return None, None, loss


class FeaturesDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, i):
            x = self.x[i]
            y = self.y[i]
            return x, y
    
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)





# linear_eval_train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2))], p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# linear_eval_test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# linear_eval_train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=linear_eval_train_transform)
# linear_eval_test_dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=linear_eval_test_transform)


# def linear_evaluation(encoder, num_epochs=10):

#     train_loader = DataLoader(linear_eval_train_dataset, batch_size=1024, shuffle=True)
#     test_loader = DataLoader(linear_eval_test_dataset, batch_size=1024, shuffle=False)


#     def extract_features(data_loader, encoder, epochs):
#         features, labels = [], []
#         # for images, labels_batch in tqdm(data_loader, desc='[Linear Eval][Features extraction]'):
#         for epoch in tqdm(range(epochs)):
#             for images, labels_batch in data_loader:
#                 with torch.no_grad():
#                     features_batch, projections_batch = encoder(images.to(device))
#                 features.append(features_batch)
#                 labels.append(labels_batch)
#         return torch.cat(features, dim=0), torch.cat(labels, dim=0)

#     # Extract features for linear evaluation
#     train_features, train_labels = extract_features(train_loader, encoder, epochs=1)
#     test_features, test_labels = extract_features(test_loader, encoder, epochs=1)

    
#     features_train_dataset = FeaturesDataset(train_features, train_labels)
#     features_test_dataset = FeaturesDataset(test_features, test_labels)

#     features_train_dataloader = DataLoader(features_train_dataset, batch_size=1024, shuffle=True)
#     features_test_dataloader = DataLoader(features_test_dataset, batch_size=1024, shuffle=True)
    
    
    
#     linear_eval_model = LinearClassifier(encoder.feature_dim, num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(linear_eval_model.parameters(), lr=0.01, momentum=0.9)
    
#     print("[Linear Eval][Training]")
#     # for epoch in tqdm(range(num_epochs), desc="[Linear Eval][Training]"):
#     for epoch in range(num_epochs):
#         linear_eval_model.train()

#         for features, labels in features_train_dataloader:
#             features, labels = features.to(device), labels.to(device)
#             outputs = linear_eval_model(features)
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
            
            
#     linear_eval_model.eval()
    
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             print("[Linear Eval][Train Eval]")
#             # for features, labels in tqdm(features_train_dataloader, desc="[Linear Eval][Train Eval]"):
#             for features, labels in features_train_dataloader:
#                 features, labels = features.to(device), labels.to(device)
#                 outputs = linear_eval_model(features)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         train_accuracy = (correct / total) * 100
        
        
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             print("[Linear Eval][Test Eval]")
#             # for features, labels in tqdm(features_test_dataloader, desc="[Linear Eval][Test Eval]"):
#             for features, labels in features_test_dataloader:
#                 features, labels = features.to(device), labels.to(device)
#                 outputs = linear_eval_model(features)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         test_accuracy = (correct / total) * 100
    
#     return train_accuracy, test_accuracy
            
            
            
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, targets=None):
    if not targets:
        targets = memory_data_loader.dataset.targets
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            
            total_num += data.size(0)

            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    feature and feature_bank are normalized
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # print(one_hot_label)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels         


def knn_evaluation(encoder):
    
    # simple_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])

    # linear_eval_train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True,  download=True, transform=simple_transform)
    # linear_eval_test_dataset  = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=simple_transform)
    
    # train_loader = DataLoader(linear_eval_train_dataset, batch_size=1024, shuffle=False)
    # test_loader = DataLoader(linear_eval_test_dataset, batch_size=1024, shuffle=False)
    
    memory_loader = get_essl_memory_loader()
    test_loader = get_essl_test_loader()
    
    acc = knn_monitor(
        encoder.enc,
        memory_loader,
        test_loader,
    )
    
    return acc


def top_k_accuracy(sim, k):
    n_samples = sim.shape[0] // 2
    sim[range(sim.shape[0]), range(sim.shape[0])] = -1
    y_index = torch.tensor(list(range(n_samples, sim.shape[0]))).reshape(-1, 1)
    acc = (sim.argsort()[:n_samples, -k:].detach().cpu() == y_index).any(-1).sum() / n_samples
    return acc.item()