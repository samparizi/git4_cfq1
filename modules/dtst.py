import h5py
import numpy as np
import torch
import torch.utils.data
from netCDF4 import Dataset
import pandas as pd
import pickle as pkl
import torchvision.transforms as t


class DTST:

    def __init__(self, data_type=1):
        super(DTST, self).__init__()
        self.data_type = data_type

    def h5py_sandeep(self):
        hpfile = '/Users/mostafa/OneDrive - alumni.ubc.ca/datas/P_train_pc.h5'
        hf = h5py.File(hpfile, 'r')
        # print(hf.keys())
        hf0 = hf['dataset'][:]
        hf0 = hf0[:, 0, :, :]

        # hf0 = torch.tensor(np.array(hf0))
        hf0 = torch.Tensor(hf0)

        print('hf0_shape_data', hf0.shape)
        return hf0

    def h5py_myfile(self):
        # hpfile = '/Users/mostafa/Dropbox/progs/datas/myfile.h5'
        hpfile = '/Users/mostafa/OneDrive - alumni.ubc.ca/datas/myfile.h5'

        hf = h5py.File(hpfile, 'r')
        # print(hf.keys())
        hf0 = hf['DS3'][:]

        hf0 = np.array(hf0).astype('float')

        # # removing nan values
        # df_shape = hf0.shape
        # hf0 = hf0.reshape(df_shape[0] * df_shape[1], df_shape[2])
        # hf0 = pd.DataFrame(hf0)
        #
        # hf0 = hf0.fillna(hf_mean)
        # # hf0 = (hf0 - hf_min) / (hf_max - hf_min)
        # hf0 = hf0.to_numpy().reshape(*df_shape)

        for i in range(hf0.shape[2]):
            hf0_mean = np.nanmean(hf0[:, :, i])
            hf0[np.isnan(hf0[:, :, i])] = hf0_mean

            hf_max = np.nanmax(hf0[:, :, i])
            hf_min = np.nanmin(hf0[:, :, i])

            hf0[:, :, i] = (hf0[:, :, i] - hf_min) / (hf_max - hf_min)

        # hf0 = torch.tensor(np.array(hf0))
        hf0 = torch.Tensor(hf0)
        hf0 = hf0.permute(2, 0, 1)

        print('hf0_shape_data', hf0.shape)
        return hf0

    def nc_file(self):
        ncfile = '/Users/mostafa/OneDrive - alumni.ubc.ca/datas/nnx2.nc'
        nf = Dataset(ncfile, mode='r')
        nf0 = nf.variables['thetao'][:]

        nf0 = np.array(nf0)

        # nf0 = torch.tensor(np.array(nf0))
        nf0 = torch.Tensor(nf0)
        nf0 = torch.squeeze(nf0)

        print('nf0_shape_data', nf0.shape)
        return nf0

    def test_dtst(self, data_type):
        if data_type == 'h5py_myfile':
            print('h5py_myfile dataset is in process')
            data = self.h5py_myfile()
        elif data_type == 'h5py_sandeep':
            print('h5py_sadeep dataset is in process')
            data = self.h5py_sandeep()
        elif data_type == 'nc_file':
            print('ncfile dataset is in process')
            data = self.nc_file()
        else:
            print("no dataset is in progress")
            data = {}

        output = open('/Users/mostafa/OneDrive - alumni.ubc.ca/datas/train/data_1.pkl', 'wb')

        pkl.dump(data, output)
        output.close()


if __name__ == "__main__":
    p = DTST(data_type=1)
