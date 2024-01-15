import torch
import torch.nn as nn


class MyModel(torch.nn.Module):
    def __init__(self, num_classes, hidden_size, num_lstm=2):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.Relu1 = nn.ReLU(inplace=True)
        self.Convadd = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.MaxPooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.MaxPooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        self.Pooling3 = nn.AdaptiveMaxPool2d((2, 2))
        self.LSTM = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_lstm, batch_first=True)
        self.FC = nn.Linear(hidden_size, num_classes)
        self.Conv = nn.Sequential(
            self.Conv1, self.Relu1, self.MaxPooling1, self.Conv2,
            self.Relu1, self.MaxPooling2, self.Conv3, self.Relu1, self.Pooling3
        )
        self.Conv_1 = nn.Sequential(
            self.Conv1, self.Relu1, self.Convadd, self.Relu1, self.MaxPooling1, self.Conv2,
            self.Relu1, self.MaxPooling2, self.Conv3, self.Relu1, self.Pooling3
        )
        self.Conv_2 = nn.Sequential(
            self.Conv1, self.Relu1, self.MaxPooling1, self.Conv3,
            self.Relu1, self.Pooling3
        )

    def forward(self, x):
        x1, x2, x3, x4, x5 = x.shape
        x = torch.reshape(x, (-1, *x.shape[2:]))
        x = self.Conv(x)
        # x = self.Conv_1(x)
        # x = self.Conv_2(x)
        x = nn.Flatten()(x)
        x = torch.reshape(x, (x1, x2, -1))
        x, (h_n, c_n) = self.LSTM(x)
        x = h_n[-1, ...]
        x = self.FC(x)
        return x

