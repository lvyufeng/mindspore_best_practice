import argparse
import mindspore.nn as nn
from src.model import Linear
from src.dataset import create_dataset

def args_parser():
    arg_parser = argparse.ArgumentParser(description="run linear regression")
    arg_parser.add_argument("--epochs", type=int, default=1, help="Epoch number, default is 1")
    arg_parser.add_argument("--num_data", type=int, default=16000, \
                            help="Number of generated dataset, default is 16000")
    arg_parser.add_argument("--batch_size", type=int, default=16, help="Batch size, default is 16")
    arg_parser.add_argument("--repeat_size", type=int, default=1, help="Repeat size of dataset, default is 1")
    args_opt = arg_parser.parse_args()
    
    return args_opt
def train_one_epoch(trainer, dataset, every_print=1, epoch_num=0):
    """train network in one epoch"""
    steps = 0
    for x, y in dataset.create_tuple_iterator():
        steps += 1
        loss = trainer(x, y)
        if steps % every_print == 0:
            print(f"epoch: {epoch_num}, loss: {loss.asnumpy()}")
    
def train(args):
    """train linear regression network"""
    # create dataset
    train_dataset = create_dataset(args.num_data, args.batch_size, args.repeat_size)
    # instantiate network
    net = Linear()
    # instantiate loss function
    loss = nn.MSELoss()
    # instantiate optimizer
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    # connect network with loss(to construct a complete computing graph)
    net_with_loss = nn.WithLossCell(net, loss)
    # connect network with optimizer(to construct a complete computing graph)
    trainer = nn.TrainOneStepCell(net_with_loss, optimizer)
    # train network
    for epoch_num in range(args.epochs):
        train_one_epoch(trainer, train_dataset, every_print=10, epoch_num=epoch_num)

if __name__ == "__main__":
    args = args_parser()
    train(args)