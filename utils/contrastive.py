import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device


     
class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()


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
    






linear_eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

linear_eval_train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=linear_eval_transform)
linear_eval_test_dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=linear_eval_transform)


def linear_evaluation(encoder):
    
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    

        
    train_loader = DataLoader(linear_eval_train_dataset, batch_size=256, shuffle=True)

    test_loader = DataLoader(linear_eval_test_dataset, batch_size=256, shuffle=False)
    
    linear_eval_model = LinearClassifier(512, num_classes=10).to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_eval_model.parameters(), lr=0.01, momentum=0.9)
    
    
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




def top_k_accuracy(sim, k):
    n_samples = sim.shape[0] // 2
    sim[range(sim.shape[0]), range(sim.shape[0])] = -1
    y_index = torch.tensor(list(range(n_samples, sim.shape[0]))).reshape(-1, 1)
    acc = (sim.argsort()[:n_samples, -k:].detach().cpu() == y_index).any(-1).sum() / n_samples
    return acc.item()