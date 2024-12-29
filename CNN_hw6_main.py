import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
import os

from CNN_hw6_models import VGG, ResNet, ResNext  # Assuming you have these in `models.py`

def train(model, args):
    '''
    Model training function
    input: 
        model: neural network classifier (e.g., VGG, ResNet, ResNext)
        args: configuration
    '''
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Correct normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Correct normalization
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(log_dir=args.log_dir)


    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

        scheduler.step()

        test(model, test_loader, args, writer, epoch)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f'Model checkpoint saved to {checkpoint_path}')


def test(model, test_loader, args, writer=None, epoch=None):
    '''
    Model testing function
    input: 
        model: neural network classifier
        test_loader: DataLoader for testing dataset
        args: configuration
        writer: SummaryWriter for TensorBoard (optional)
        epoch: current epoch (optional)
    '''
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    if writer and epoch is not None:
        writer.add_scalar('Test Loss', test_loss / len(test_loader), epoch)
        writer.add_scalar('Test Accuracy', accuracy, epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training/testing')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model', type=str, choices=['VGG', 'ResNet', 'ResNext'], required=True, help='Model type to use')
    args = parser.parse_args()

    if args.model == 'VGG':
        model = VGG(num_classes=10)  # Assuming CIFAR-10 dataset
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'ResNext':
        model = ResNext(num_classes=10)

    model.to(args.device)

    train(model, args)

