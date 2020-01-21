import argparse
import modules.advecdiffus as warps

parser = argparse.ArgumentParser(description='main framework for fluid simulation')
print('implementing args')

parser.add_argument('--data_type', default='h5py_myfile', choices=('h5py_myfile', 'h5py_sandeep', 'nc_file'),
                    help='choose from (h5py_myfile, h5py_sandeep, nc_file)')
parser.add_argument('--env', default='main',
                    help='environnment for visdom_main or max_epoch=100')

parser.add_argument('--train_root', metavar='DIR', default='/Users/mostafa/Desktop/datas/train/',
                    help='path to training dataset')
parser.add_argument('--test-root', metavar='DIR', default='/Users/mostafa/Desktop/datas/train/',
                    help='path to testing dataset')

parser.add_argument('--seq-len', default=4, type=int,
                    help='number of input images as input of the estimator (horizon)')
parser.add_argument('--target-seq-len', default=6, type=int,
                    help='number of target images')

parser.add_argument('-s', '--split', default=.8, type=float, metavar='%',
                    help='split percentage of train samples vs test (default: .8)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 16 or 64?)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--upsample', default='bilinear', choices=('deconv', 'nearest', 'bilinear'),
                    help='choose from (deconv, nearest, bilinear)')

warp_names = sorted(name for name in warps.__dict__
                    if not name.startswith('__'))

parser.add_argument('--warp', default='BilinearWarpingScheme', choices=warp_names,
                    help='choose warping scheme to use:' + ' | '.join(warp_names))

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay (default: 4e-4)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200,300,500?')

parser.add_argument('--smooth-coef', default=0.4, type=float,
                    help='coefficient associated to smoothness loss in cost function')
parser.add_argument('--div-coef', default=1, type=float,
                    help='coefficient associated to divergence loss in cost function')
parser.add_argument('--magn-coef', default=-0.003, type=float,
                    help='coefficient associated to magnitude loss in cost function')

parser.add_argument('--no-plot', action='store_true',
                    help='no plot images using visdom')
