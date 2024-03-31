import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import deeplake
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define model training and evaluation function
def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, writer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['images'], data['labels']
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.flatten())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)

        # Evaluation
        model.eval()
        evaluate_model(model, train_loader, writer, 'Train', epoch)
        evaluate_model(model, test_loader, writer, 'Test', epoch)

def evaluate_model(model, data_loader, writer, prefix, epoch):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data['images'], data['labels']
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.flatten().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    confusion_matrix_data = confusion_matrix(all_labels, all_preds)

    writer.add_scalar(f'{prefix} Accuracy', accuracy, epoch)
    writer.add_scalar(f'{prefix} Precision', precision, epoch)
    writer.add_scalar(f'{prefix} Recall', recall, epoch)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', ax=ax)
    writer.add_figure(f'{prefix} Confusion Matrix', fig, global_step=epoch)

# Define your custom MLP model
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(16*16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 16*16)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define your custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*4*4)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load USPS dataset training and testing subsets
ds_train = deeplake.load('hub://activeloop/usps-train')
ds_test = deeplake.load('hub://activeloop/usps-test')

# Create PyTorch dataloaders
train_loader = ds_train.pytorch(num_workers=0, batch_size=64, shuffle=True)
test_loader = ds_test.pytorch(num_workers=0, batch_size=64, shuffle=False)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Model configurations
configurations = [
    {'lr': 0.001},  # Configuration 1: Learning rate = 0.001
    {'lr': 0.01},   # Configuration 2: Learning rate = 0.01
    {'lr': 0.0001}  # Configuration 3: Learning rate = 0.0001
]

# Train and evaluate MLP model with different configurations
for i, config in enumerate(configurations, 1):
    # Initialize the MLP model
    mlp_model = CustomMLP()

    # Define loss function and optimizer for MLP
    mlp_criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=config['lr'])

    # Train and evaluate MLP model
    train_and_evaluate_model(mlp_model, train_loader, test_loader, mlp_criterion, mlp_optimizer, writer)

# Train and evaluate CNN model with different configurations
for i, config in enumerate(configurations, 1):
    # Initialize the CNN model
    cnn_model = CustomCNN()

    # Define loss function and optimizer for CNN
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=config['lr'])

    # Train and evaluate CNN model
    train_and_evaluate_model(cnn_model, train_loader, test_loader, cnn_criterion, cnn_optimizer, writer)

# Close TensorBoard writer
writer.close()

print('Finished Training and Evaluation')
