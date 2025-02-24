#MLP
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 #28*28
hidden_size = 500
num_classes = 10
num_epoch = 2
batch_size = 100
learning_rate = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root = './data', train = True,
                                           transform = transforms.ToTensor(), download = True)

test_dataset = torchvision.datasets.MNIST(root = './data', train = False,
                                           transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size, shuffle= False)

example  = iter(train_loader)
samples, labels = next(example)
print(samples.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap = 'gray')
#plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size ,hidden_size, num_classes).to(device)

#loss ans optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_step = len(train_loader)
for epoch in range (num_epoch):
    for i,(images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to (device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 ==0:
            print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_total_step}, loss = {loss.item():.4f}')
#test
with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        #value ,index
        _, prediction = torch.max(outputs.data, 1)
        n_sample += labels.shape[0]
        n_correct += (prediction == labels).sum().item()

    acc = 100.0 * n_correct / n_sample
    print(f'accuracy = {acc} %')