import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


class KidneyDiseaseModelAF(nn.Module):
    def __init__(self, input_size=24, output_size=1):
        super(KidneyDiseaseModelAF, self).__init__()
        self.norm0 = NormFunction1D(input_size)
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()

        self.norm1 = NormFunction1D(64)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.norm2 = NormFunction1D(32)
        self.fc3 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm0(x)
        out = self.fc1(x)
        out = self.relu(out)

        out = self.norm1(out)
        out = self.fc2(out)
        out = self.relu2(out)

        out = self.norm2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class KidneyDiseasePredictorAF:
    def __init__(self, model_path):
        self.model = KidneyDiseaseModelAF()
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
                predicted = (outputs > 0.5).float()
                y_pred.extend(predicted.numpy())

        return y_pred

    def predict_proba(self, X):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0]
                outputs = self.model(inputs)
                y_pred.extend(outputs.numpy())

        return y_pred
