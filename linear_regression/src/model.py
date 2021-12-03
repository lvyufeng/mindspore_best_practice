import mindspore
import mindspore.nn as nn

class Linear(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__()
        self.linear = nn.Dense(1, 1)

    def construct(self, inputs):
        return self.linear(inputs)