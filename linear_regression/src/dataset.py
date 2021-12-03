import numpy as np
from mindspore.dataset import GeneratorDataset

def get_data(num, w=2.0, b=3.0):
    X, Y = [], []
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        X.append(np.array([x]).astype(np.float32))
        Y.append(np.array([y]).astype(np.float32))
    return X, Y

class Data(object):
    def __init__(self, num_data):
        self.X, self.Y = get_data(num_data)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = GeneratorDataset(Data(num_data), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data