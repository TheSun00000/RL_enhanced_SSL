import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm

from utils.networks import build_resnet50


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device

linear_eval_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2))], p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

linear_eval_test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

linear_eval_train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=linear_eval_train_transform)
linear_eval_test_dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=linear_eval_test_transform)


class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.feature_dim, num_classes, bias=True)

    def forward(self, x):
        x, _ = self.encoder(x)
        return self.fc(x)


def train_val(net, data_loader, train_optimizer, epoch, epochs):
        is_train = train_optimizer is not None
        net.train() if is_train else net.eval()
        
        loss_criterion = nn.CrossEntropyLoss()

        total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
        with (torch.enable_grad() if is_train else torch.no_grad()):
            for data, target in data_bar:
                data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
                out = net(data)
                loss = loss_criterion(out, target)

                if is_train:
                    train_optimizer.zero_grad()
                    loss.backward()
                    train_optimizer.step()

                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                
                data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                        .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                                total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100



def linear_evaluation(encoder, epochs=100):

    train_loader = DataLoader(linear_eval_train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(linear_eval_test_dataset, batch_size=512, shuffle=False)
    
    
    linear_eval_model = LinearClassifier(encoder, num_classes=10).to(device)
    for param in linear_eval_model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.SGD(linear_eval_model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        train_loss, train_acc_1, train_acc_5 = train_val(linear_eval_model, train_loader, optimizer, epoch, epochs)
        if epoch % 5 == 0:
            test_loss, test_acc_1, test_acc_5 = train_val(linear_eval_model, test_loader, None, epoch, epochs)
    
    return













encoder = build_resnet50()
encoder.load_state_dict(torch.load('params/params_192/encoder.pt'))
encoder = encoder.to(device)


linear_evaluation(encoder)