import torch.nn as nn
import torch
import timm


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_lstm_layers=2, backbone_name='resnet101'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.adap = nn.AdaptiveAvgPool2d((2, 2))

        self.lstm = nn.LSTM(2048, hidden_size, num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        'x: batch, num_frames, channels, height, width'
        batch, num_frames, channels, height, width = x.shape

        # x: batch * num_frames, channels, height, width
        x = torch.reshape(x, (-1, *x.shape[2:]))

        x1, x2, x3, x4, x5 = self.backbone(x)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)

        x = self.adap(x3)

        x = nn.Flatten()(x)

        x = torch.reshape(x, (batch, num_frames, -1))

        x, (h_n, c_n) = self.lstm(x)

        x = h_n[-1, ...]

        x = self.fc(x)

        return x
