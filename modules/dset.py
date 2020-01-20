import pickle as pkl
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch

# import modules.args_cfq1
# args = modules.args_cfq1.parser.parse_args()


class DSET(torch.utils.data.Dataset):

    def __init__(self, root=0, seq_len=4, target_seq_len=6,
                 transform=True,
                 zones=None,
                 ):
        # super(DSET, self).__init__()

        if zones is None:  # using all zones
            zones = range(1, 2)

        self.root = root
        self.zones = zones
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.transform = transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop(10),
            transforms.Resize(size=(60, 60)),
            # transforms.Resize(60),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data = {}

        for zone in zones:
            self.data[zone] = pkl.load(open('/Users/mostafa/OneDrive - alumni.ubc.ca/datas/train/data_1.pkl', 'rb'))

        self.num_single = self.data[zones[0]].shape[0] - self.seq_len - self.target_seq_len - 1  # ( # -1 ?)
        self.num = self.num_single * len(zones)

    def test_dset(self):
        pass

    def __getitem__(self, index):

        zone = self.zones[index // (self.num_single - 1)]
        sample_num = index % (self.num_single - 1)
        pdata = self.data[zone]

        input = pdata[sample_num: sample_num + self.seq_len, :, :]
        target = pdata[sample_num + self.seq_len: sample_num + self.seq_len + self.target_seq_len, :, :]

        return self.transform(input), self.transform(target)
        # return input, target

    def __len__(self):
        return self.num - len(self.zones)
