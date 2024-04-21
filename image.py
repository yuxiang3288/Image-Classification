from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define your dataset path and transformations
dataset_path = 'C:\\Users\\choo3\\Desktop\\image_classification\\dataset'  # Replace with the path to your dataset folder

# dataset_path = 'path\\to\\dataset'  # Replace with the path to your dataset folder


transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),  # Convert to tensor
])

# Load your dataset with ImageFolder, which expects subfolders for each class
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Since you have a very small dataset, we'll just simulate train, val, test with the same images
indices = list(range(len(full_dataset)))  # Assume we only have 10 images
# Create subsets based on the indices
train_indices = indices[:20]
val_indices = indices[20:24]
test_indices = indices[24:28]

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)
test_subset = Subset(full_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=2, shuffle=False)

# Define function to show data information and images
def show_data(loader, title):
    print(f"Showing information and images for: {title}")
    images, labels = next(iter(loader))  # Get a batch of images and labels
    print(f"Number of data entries: {len(images)}")
    print(f"Number of classes: {len(np.unique(labels.numpy()))}")
    print(f"Shape of the image size: {images[0].size()}")

    # Plotting images with corresponding labels
    plt.figure(figsize=(15, 3))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Display information for training set and plot images
show_data(train_loader, "Training Set")

# Assuming the Network class and the dataloaders (train_loader and val_loader) are already defined
class Network(nn.Module):
    def __init__(self, num_classes=2):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * 32 * 32, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 32 * 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
# Initialize the network
num_classes = 2  # example for a binary classification
network = Network(num_classes=num_classes)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (Let's use Adam optimizer as an example)
optimizer = optim.Adam(network.parameters(), lr=0.001)

# Number of epochs
num_epochs = 50

# Lists for storing loss and accuracy values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    network.train()  # set network to training phase
    running_loss = 0.0
    correct = 0
    total = 0

    # Training step
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = network(inputs)
        # print(f"Output shape: {outputs.shape}")
        # print(f"Labels shape: {labels.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # Validation step
    network.eval()  # set network to evaluation phase
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_losses.append(running_loss / len(val_loader))
    val_accuracies.append(100 * correct / total)

    # Print statistics
    print(f"Epoch {epoch+1}/{num_epochs}.. "
          f"Train loss: {train_losses[-1]:.3f}.. "
          f"Train accuracy: {train_accuracies[-1]:.3f}%.. "
          f"Val loss: {val_losses[-1]:.3f}.. "
          f"Val accuracy: {val_accuracies[-1]:.3f}%")

# Check if the model has converged by looking at the trend of the loss and accuracy
# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(val_accuracies, label='Validation accuracy')
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Put the model in evaluation mode
network.eval()

# Initialize lists to track the labels and predictions
true_labels = []
predictions = []

# Disable gradient computation since we are in inference mode
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass to get the outputs
        # The outputs are a series of class probabilities
        outputs = network(images)
        
        # Get the predicted class with the highest probability
        _, predicted = torch.max(outputs, 1)
        
        # Extend our lists
        true_labels.extend(labels.numpy())
        predictions.extend(predicted.numpy())

# Calculate accuracy manually
accuracy = sum(t == p for t, p in zip(true_labels, predictions)) / len(true_labels)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix manually
num_classes = 2  # Change this to the number of classes you have
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(true_labels, predictions):
    conf_matrix[t, p] += 1

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)