import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse



class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int = 3 * 32 * 32, out_channels: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.fc(x)


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=16*16*128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [batch_size, 64, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 128, 16, 16]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)   # Flatten to [batch_size, 8*8*128]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def train(model, optimizer, scheduler, args):
    loss_function = nn.CrossEntropyLoss()

    # Create datasets and dataloaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(trainloader):.4f}')
        test(model, testloader, args)

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, f'checkpoint_epoch_{epoch+1}.pth')

def test(model, testloader, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier()
    elif args.model == 'fcnn':
        model = FCNN()
    else:
        raise AssertionError(f"Model {args.model} not recognized")

    # create optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        raise AssertionError(f"Optimizer {args.optimizer} not recognized")

    # create scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        raise AssertionError(f"Scheduler {args.scheduler} not recognized")

    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else:
        raise AssertionError(f"Run option {args.run} not recognized")
