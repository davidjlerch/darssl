import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, clas_dim, dropout=0.):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, clas_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


class StackedModel(nn.Module):
    def __init__(self, model1, model2):
        super(StackedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, src, total_frames, memory_type):
        mem, out = self.model1(src, total_frames, memory_type)
        cls_in = mem[:, 0, :]
        x__ = self.model2(cls_in)
        return x__
