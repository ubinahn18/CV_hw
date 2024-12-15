import torch
import torch.nn as nn
import argparse


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int = 3 * 32 * 32, out_channels: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.fc(x)


class FCNN(nn.Module):
    def __init__(self, in_channels: int = 3 * 32 * 32, hidden_channels: int = 512, out_channels: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.model(x)



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
        model = LinearModel()
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
