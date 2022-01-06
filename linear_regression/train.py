import argparse
import mindspore.nn as nn
from src.model import Linear
from src.dataset import create_dataset
from easy_mindspore import value_and_grad
from mindspore import ms_function

def args_parser():
    arg_parser = argparse.ArgumentParser(description="run linear regression")
    arg_parser.add_argument("--epochs", type=int, default=1, help="Epoch number, default is 1")
    arg_parser.add_argument("--num_data", type=int, default=16000, \
                            help="Number of generated dataset, default is 16000")
    arg_parser.add_argument("--batch_size", type=int, default=16, help="Batch size, default is 16")
    arg_parser.add_argument("--repeat_size", type=int, default=1, help="Repeat size of dataset, default is 1")
    args_opt = arg_parser.parse_args()
    
    return args_opt

def train_one_epoch(net, loss_fn, optimizer, dataset, every_print=1, epoch_num=0):
    """train network in one epoch"""
    @ms_function
    def train_step(x, y):
        logits = net(x)
        loss = loss_fn(logits, y)
        return loss, logits

    grad_fn = value_and_grad(train_step, net.trainable_params(), has_aux=True)
    steps = 0
    for x, y in dataset.create_tuple_iterator():
        steps += 1
        (loss, _), grads = grad_fn(x, y)
        optimizer(grads)
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
    # train network
    for epoch_num in range(args.epochs):
        train_one_epoch(net, loss, optimizer, train_dataset, every_print=10, epoch_num=epoch_num)

if __name__ == "__main__":
    args = args_parser()
    train(args)