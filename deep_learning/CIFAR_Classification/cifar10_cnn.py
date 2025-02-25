import torch 
import torch.nn.functional as f
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib as plt

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
num_epoch = 1
batch_size = 4
learning_rate = 0.001

#transform image to tensor and normalize them
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data1',train =True,
                                             download =True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data1',train =False,
                                             download =True, transform = transform)

#pour faciliter le training nous avons divisé nos donnée en des petits batch
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
            batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
            batch_size = batch_size, shuffle = False)

classes = ('plane','car','bird','cat',
    'deer','dog','frog','horse','ship','truck')

# implimentation of the convolutional network

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)   #Réduit la taille des features maps 
                                        #en prenant le maximum sur des régions de 2×2

        self.conv2 = nn.Conv2d(6,16,5)  #on flatten les features maps obtenue à cette deuxième
                                        #convolution de les envoyer aux fully connected layer 
                                        # avant de passer à la couche suivante
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)    #output size de 10 en sortie vu qu'on prédit 10 classes.

# puis nous avons définie notre pipeline d'entrainement qui consiste à faire du foward pass
    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

model = ConvNet().to(device)
criterion =nn.CrossEntropyLoss() #qui prend déja en compte la softmax qui une fonction d'activation
                                 #permetant de faires des classifications multi classes
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_dataset)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
#parler du criterion
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward() #nous calculons le graidient de la loss par rapport aux poids et aux biais
                        #par rétropropagation en utilisant la chain rule
        optimizer.step() #nous mettons à jours ces poids grace à notre opimizer Stochastic Gradient Descent (SGD)

        if (i+1)% 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}, Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}]')
print('Finished Training')
"""
with torch.no_grad():
    n_correct = 0
    n_sample = 0
    n_class_correct = [0 for i in range (10)]
    n_class_samples = [0 for i in range (10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #max return value
        _,predicted = torch.max(outputs, 1)
        n_sample += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            labels = labels[i]
            pred = predicted[i]
            if (labels == pred):
                n_class_correct[labels] +=1
            n_class_correct[labels] +=1

        acc = 100.0 * n_correct / n_sample
        print(f'Accuracy of the network: {acc}%')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_correct[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
            
"""
# nous avons évalué notre model sur le test set et nous avons obtenue une accuracy de 95%
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')