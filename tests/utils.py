from torch import nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(3072, 10))

    def forward(self, x):
        return self.main(x)
