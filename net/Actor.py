from torch import nn
import torch
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.h0 = None
        self.c0 = None

    def forward(self, x):

        if self.h0 is None or self.h0.size(1) != x.size(0) or self.h0.device != x.device:
            self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (self.h0, self.c0) = self.lstm(x, (self.h0.detach(), self.c0.detach()))
        self.h0, self.c0 = self.h0.detach(), self.c0.detach()
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)

