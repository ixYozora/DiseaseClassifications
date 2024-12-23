import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""

@author: Iraj Masoudian
"""


class KidneyDiseaseModel(nn.Module):
    def __init__(self, input_size=9, output_size=1):
        super(KidneyDiseaseModel, self).__init__()
        self.norm0 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()

        self.norm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.norm2 = nn.BatchNorm1d(32)
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


class KidneyDiseasePredictor:
    def __init__(self, model_path):
        self.model = KidneyDiseaseModel()
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
