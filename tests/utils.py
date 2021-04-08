from torch import nn


class DummyModel(nn.Module):
    def __init__(self, in_shape=3072, out_shape=10):
        super().__init__()
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(in_shape, out_shape, bias=False))

    def forward(self, x):
        return self.main(x)
