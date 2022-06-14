import torch.nn as nn
import math


class BaselineModel(nn.Module):
    """
    ToDo
    """
    def __init__(self, input_size, n_classes, p_dropout=0.3):
        super(BaselineModel, self).__init__()

        self.neural_net = nn.Sequential(
            nn.Linear(input_size, math.floor(input_size / 2)),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(math.floor(input_size / 2), math.floor(input_size / 3))
        )

        self.classification_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(math.floor(input_size / 3), n_classes)
        )

    def neural_net_forward(self, x):
        return self.neural_net(x)

    def classification_forward(self, encoded_x):
        return self.classification_layer(encoded_x)
