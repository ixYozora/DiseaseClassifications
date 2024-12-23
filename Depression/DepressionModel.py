import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

"""

@author: Iraj Masoudian
"""


class NormFunction1D(nn.Module):
    def __init__(self, features):
        super(NormFunction1D, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones(1, features))
        self.beta = torch.nn.Parameter(torch.zeros(1, features))
        self.eps = 0.00001

        self.gamma.requires_grad = True
        self.beta.requires_grad = True

        self.m = torch.nn.Parameter(torch.ones(1, features))
        self.v = torch.nn.Parameter(torch.ones(1, features))

        self.m.requires_grad = True
        self.v.requires_grad = True

        self.features = features

    def forward(self, x):
        x = torch.add(torch.mul(torch.div(torch.subtract(x, self.m), torch.sqrt(self.v + self.eps)), self.gamma),
                      self.beta)
        return x


class DepressionModel(nn.Module):
    def __init__(self, input_size=11, num_classes=4):
        super(DepressionModel, self).__init__()
        self.norm0 = NormFunction1D(input_size)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm0(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities


class DepressionPredictor:
    def __init__(self, model_path):
        self.model = DepressionModel()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, X):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0]
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.numpy())
        return y_pred

    def predict_proba(self, X):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        y_prob = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0]
                probabilities = self.model.predict_proba(inputs)
                y_prob.extend(probabilities.numpy())
        return y_prob


