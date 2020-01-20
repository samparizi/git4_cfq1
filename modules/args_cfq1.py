import argparse

parser = argparse.ArgumentParser(description='main framework for fluid simulation')
print('implementing args')

parser.add_argument('--data_type', default='h5py_myfile', choices=('h5py_myfile', 'h5py_sandeep', 'nc_file'),
                    help='choose from (h5py_myfile, h5py_sandeep, nc_file)')
parser.add_argument('--env', default='main',
                    help='environnment for visdom_main or max_epoch=100')
parser.add_argument('--train_root', metavar='DIR', default='/Users/mostafa/Desktop/datas/train/',
                    help='path to training dataset')
parser.add_argument('--seq-len', default=4, type=int,
                    help='number of input images as input of the estimator (horizon)')
parser.add_argument('--target-seq-len', default=6, type=int,
                    help='number of target images')
