import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Function for loading notMNIST Dataset
def loadData(datafile = "notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Custom Dataset class.
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define CNN
class CNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(drop_out_p)
        self.fc1 = nn.Linear(64*4*4, 784)
        self.fc2 = nn.Linear(784, 10)
        

    def forward(self, x):
        x = self.pool1(self.norm1(F.relu(self.conv1(x)))) #not sure if this is correct
        x = self.pool2(self.norm2(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Define FNN
class FNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(FNN, self).__init__()
        #TODO
        #DEFINE YOUR LAYERS HERE
        self.fc1 = nn.Linear(28*28, 10)
        self.fc2 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(drop_out_p)
        self.fc3 = nn.Linear(10, 10)
        

    def forward(self, x):
        #TODO
        #DEFINE YOUR FORWARD FUNCTION HERE
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
        

# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):

    model.eval()
    device = next(model.parameters()).device
    accuracy = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # TODO
            # Return the accuracy
            output = model(images)
            prediction = torch.argmax(output, 1)
            correct += (prediction == labels).sum().item() 
            total += labels.size(0)  
        accuracy = correct/total * 100
            

    return accuracy

def train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=50, verbose=False):
  #TODO
  # Define your cross entropy loss function here
  # Use cross entropy loss
  criterion = torch.nn.CrossEntropyLoss()
  
  #TODO
  # Define your optimizer here
  # Use AdamW optimizer, set the weights, learning rate and weight decay argument.
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  acc_hist = {'train':[], 'val':[], 'test': []}

  for epoch in range(num_epochs):
    model = model.train()
    ## training step
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # TODO
        # Follow the step in the tutorial
        ## forward + backprop + loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## update model params
        

    model.eval()
    acc_hist['train'].append(get_accuracy(model, train_loader))
    acc_hist['val'].append(get_accuracy(model, val_loader))
    acc_hist['test'].append(get_accuracy(model, test_loader))

    if verbose:
      print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f' \
           %(epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1]))

  return model, acc_hist

def experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.5, weight_decay=0.01, num_epochs=50, verbose=False):
  # Use GPU if it is available.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Inpute Batch size:
  BATCH_SIZE = 32

  # Convert images to tensor
  transform = transforms.Compose(
      [transforms.ToTensor()])

  # Get train, validation and test data loader.
  trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

  train_data = notMNIST(trainTarget, trainData, transform=transform)
  val_data = notMNIST(validTarget, validData, transform=transform)
  test_data = notMNIST(testTarget, testData, transform=transform)


  train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

  # Specify which model to use
  if model_type == 'CNN':
    model = CNN(dropout_rate)
  elif model_type == 'FNN':
    model = FNN(dropout_rate)


  # Loading model into device
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  model, acc_hist = train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=num_epochs, verbose=verbose)

  # Release the model from the GPU (else the memory wont hold up)
  model.cpu()

  return model, acc_hist

def compare_arch():
    fnn, fnn_hist = experiment(model_type='FNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0)
    cnn, cnn_hist = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0)
    
    epochs = range(50)
   
    plt.figure()
    plt.plot(epochs, fnn_hist['train'], label='FNN Train')
    plt.plot(epochs, cnn_hist['train'], label='CNN Train')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy %')
    plt.title('FNN vs CNN Training Acc')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, fnn_hist['test'], label='FNN Test')
    plt.plot(epochs, cnn_hist['test'], label='CNN Test')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy %')
    plt.title('FNN vs CNN Testing Acc')
    plt.legend()
    plt.show()


def compare_dropout():
    
    dropouts = [0.5, 0.8, 0.95]
    acc_histories = []
    for dropout in dropouts:
        _, hist = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=dropout, weight_decay=0.0)
        acc_histories.append(hist)
    epoches = range(50)
    plt.figure()
    for d, hist in zip(dropouts, acc_histories):
        plt.plot(epoches, hist['train'], label=f'Dropout={d} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('CNN Training Accuracy with Different Dropout Rates')
    plt.legend()
    plt.show()

    plt.figure()
    for d, hist in zip(dropouts, acc_histories):
        plt.plot(epoches, hist['test'], label=f'Dropout={d} Test')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy (%)')
    plt.title('CNN Testing Accuracy with Different Dropout Rates')
    plt.legend()
    plt.show()
    
def compare_l2():

    decays = [0.1, 1.0, 10.0]
    acc_histories = []

    for decay in decays:
        _, hist = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=decay)
        acc_histories.append(hist)

    epoches = range(50)

    plt.figure()
    for w, hist in zip(decays, acc_histories):
        plt.plot(epoches, hist['train'], label=f'Weight Decay={w} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('CNN Training Accuracy with Different Weight Decays')
    plt.legend()
    plt.show()

    plt.figure()
    for w, hist in zip(decays, acc_histories):
        plt.plot(epoches, hist['test'], label=f'Weight Decay={w} Test')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy (%)')
    plt.title('CNN Testing Accuracy with Different Weight Decays')
    plt.legend()
    plt.show()
    
    
    
    


if __name__ == '__main__':
    # compare_arch()
    # compare_dropout()
    compare_l2()
    
    
    

