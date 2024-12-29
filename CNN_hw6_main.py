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
    # Create dataset and data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Create SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Log every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

        # Adjust learning rate with scheduler
        scheduler.step()

        # Test after each epoch
        test(model, test_loader, args, writer, epoch)

    # Save checkpoint
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

            # Forward
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Log metrics to TensorBoard
    if writer and epoch is not None:
        writer.add_scalar('Test Loss', test_loss / len(test_loader), epoch)
        writer.add_scalar('Test Accuracy', accuracy, epoch)


if __name__ == '__main__':
    import argparse

    # Argument parser for configurations
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training/testing')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model', type=str, choices=['VGG', 'ResNet', 'ResNext'], required=True, help='Model type to use')
    args = parser.parse_args()

    # Select model
    if args.model == 'VGG':
        model = VGG(num_classes=10)  # Assuming CIFAR-10 dataset
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'ResNext':
        model = ResNext(num_classes=10)

    model.to(args.device)

    # Train the model
    train(model, args)

