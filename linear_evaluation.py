import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)



transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits
    
    
    
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

encoder = build_resnet18()
encoder.load_state_dict(torch.load('params/resnet18_contrastive.pt', map_location=device))
encoder = encoder.to(device)



linear_eval_model = LinearClassifier(512, num_classes=10).to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(linear_eval_model.parameters(), lr=0.01, momentum=0.9)

# Train the linear evaluation model
num_epochs = 100

for epoch in range(num_epochs):
    linear_eval_model.train()

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        with torch.no_grad():
            features, projections = encoder(inputs)
        outputs = linear_eval_model(features)

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
    if (epoch+1) % 5 == 0:
        linear_eval_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                features, projections = encoder(inputs)
                outputs = linear_eval_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Linear evaluation accuracy on CIFAR-10: {accuracy * 100:.2f}%")