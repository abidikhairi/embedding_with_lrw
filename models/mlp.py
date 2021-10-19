import torch
from torch import nn

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

    def __init__(self, num_classes, feature_size, hidden_size, device = 'cpu'):

        self.mlp = MLP(feature_size=feature_size, hidden_size=hidden_size, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)
        
    @torch.no_grad()
    def evaluate(self, x_test):
        x_test = torch.tensor(x_test).to(self.device)
        logits = self.mlp(x_test)
        _, preds = torch.max(logits, dim=1)
        
        return preds.numpy()
    
    def predict(self, x_test):
        return self.evaluate(x_test)
        
    def train(self, x_train, y_train, epochs = 10, learning_rate = 0.001):    
        self.mlp.train()
        self.mlp.to(self.device)
        
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        
        for _ in range(epochs):
            optimizer.zero_grad()

            features, labels = x_train, y_train
            logits = self.mlp(features)

            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def fit(self, x_train, y_train):
        x_train = torch.tensor(x_train).to(self.device)
        y_train = torch.tensor(y_train).long().to(self.device)

        self.train(x_train, y_train)

        return self