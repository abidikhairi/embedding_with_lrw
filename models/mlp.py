import torch
import torchmetrics.functional as thmetrics
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, feature_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, feats):
        logits = self.classifier(feats)

        return logits

class MLPTrainer():

    def __init__(self, features, labels, num_classes, feature_size, hidden_size, batch_size = 128, train_ratio = 0.8, device = 'cpu'):

        self.features = features
        self.labels = labels
        self.mlp = MLP(feature_size=feature_size, hidden_size=hidden_size, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)
        self.dataset = TensorDataset(features, labels)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
    
    def loader(self):
        num_train = int(len(self.dataset) * self.train_ratio)
        num_test = len(self.dataset) - num_train
        trainset, testset = random_split(self.dataset, [num_train, num_test])
        return DataLoader(trainset, batch_size=self.batch_size, shuffle=True), DataLoader(testset, batch_size=self.batch_size, shuffle=True)

    @torch.no_grad()
    def evaluate(self, loader):
        acc = []
        for features, labels in loader:
            features, labels = features.to(self.device), labels.to(self.device)
            logits = self.mlp(features)
            acc.append(thmetrics.accuracy(logits, labels))
        
        return torch.tensor(acc).mean()

    def train(self, epochs = 10, learning_rate = 0.01):
        trainloader, testloader = self.loader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for features, labels in trainloader:
                self.mlp.train()
                optimizer.zero_grad()

                features, labels = features.to(self.device), labels.to(self.device)
                logits = self.mlp(features)

                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                accuracy = self.evaluate(testloader)

                print('Epoch: {}, Train loss: {}, Accuracy: {:.4f} %'.format(epoch, loss, accuracy))