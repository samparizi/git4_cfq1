import numpy as np
import visdom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import modules.args_cfq1
import modules.cfd as cfd
import modules.dtst as dtst
import modules.dset as dset
import modules.cnne as cnne
import modules.vaee as vaee
import modules.rbm as rbm
import modules.cnnd as cnnd
import modules.vaed as vaed
import modules.warp as warp
import modules.rslt as rslt

import modules.dencoders as endecs
import modules.advecdiffus as warps
import modules.losses as losses
from modules.meter import AverageMeters
import modules.plots as plot


class CFQ(cfd.CFD, dtst.DTST, dset.DSET, cnne.CNNE, vaee.VAEE,
          rbm.RBM, cnnd.CNND, vaed.VAED, warp.WARP, rslt.RSLT):

    def __init__(self, _cfd, _dtst, _dset, _cnne, _vaee,
                 _rbm, _cnnd, _vaed, _warp, _rslt, **kw):
        super(CFQ, self).__init__(**kw)

        self._cfd = _cfd
        self._dtst = _dtst
        self._dset = _dset
        self._cnne = _cnne
        self._vaee = _vaee
        self._rbm = _rbm
        self._cnnd = _cnnd
        self._vaed = _vaed
        self._warp = _warp
        self._rslt = _rslt

    def for_cfd(self):

        if self._cfd is None:
            print('cfd is NOT active')
            pass
        else:
            print('cfd is active')
            return cfd.CFD.test_cfd(self) + 1

    def for_dtst(self, data_type):

        if self._dtst is None:
            print('preparing the dtst is NOT active')
            pass
        else:
            print('preparing the dtst is active')
            return dtst.DTST.test_dtst(self, data_type)

    def for_dset(self, train_data, test_data):

        if self._dset is None:
            print('dset is NOT active')
        else:
            print('dset is active')

            train_indices = range(0, int(len(train_data[0]) * args.split))  # >> range(0, 1)
            val_indices = range(int(len(train_data[0]) * args.split), len(train_data[0]))  # >> range(1, 2)

            train_loader = DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      sampler=SubsetRandomSampler(train_indices),
                                      num_workers=args.workers,
                                      pin_memory=True
                                      )

            val_loader = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    sampler=SubsetRandomSampler(val_indices),
                                    num_workers=args.workers,
                                    pin_memory=True
                                    )

            test_loader = DataLoader(test_data,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     pin_memory=True
                                     )

            splits = {
                'train': train_loader,
                'valid': val_loader,
                'test': test_loader,
            }

            return train_indices, val_indices, train_loader, val_loader, test_loader, splits
            # data = dset.DSET.test_dset(self)

    def for_cnne(self, data):

        if self._cnne is None:
            print('cnne is NOT active')
            pass
        else:
            print('cnne is active')
            data = cnne.CNNE.test_cnne(self, data) + 1
        return self.for_dtst(data)

    def for_vaee(self, data):

        if self._vaee is None:
            print('vaee is NOT active')
            pass
        else:
            print('vaee is active')
            data = vaee.VAEE.test_vaee(self, data) + 1
        return self.for_cnne(data)

    def for_rbm(self, data):

        if self._rbm is None:
            print('rbm is NOT active')
            pass
        else:
            print('rbm is active')
            data = rbm.RBM.test_rbm(self, data) + 1
        return self.for_vaee(data)

    def for_cnnd(self, data):

        if self._cnnd is None:
            print('cnnd is NOT active')
            pass
        else:
            print('cnnd is active')
            data = cnnd.CNND.test_cnnd(self, data) + 1
        return self.for_rbm(data)

    def for_vaed(self, data):

        if self._vaed is None:
            print('vaed is NOT active')
            pass
        else:
            print('vaed is active')
            data = vaed.VAED.test_vaed(self, data) + 1
        return self.for_cnnd(data)

    def for_warp(self, data):

        if self._warp is None:
            print('warp is NOT active')
            pass
        else:
            print('warp is active')
            data = warp.WARP.test_warp(self, data) + 1
        return self.for_vaed(data)

    def for_loss(self, ):

        print('\n>>>> implementing loss function...\n')
        photo_loss = nn.MSELoss()
        smooth_loss = losses.SmoothnessLoss(nn.MSELoss())
        div_loss = losses.DivergenceLoss(nn.MSELoss())
        magn_loss = losses.MagnitudeLoss(nn.MSELoss())
        cudnn.benchmark = True
        optimizer = optim.Adam(endecoder.parameters(), args.lr,
                               betas=(args.momentum, args.beta),
                               weight_decay=args.weight_decay,
                               )
        return photo_loss, smooth_loss, div_loss, magn_loss, optimizer

    def for_rslt(self, data):

        if self._rslt is None:
            print('warp is NOT active')
            pass
        else:
            print('warp is active')
            data = rslt.RSLT.test_rslt(self, data) + 1
        return self.for_warp(data)


if __name__ == "__main__":
    args = modules.args_cfq1.parser.parse_args()
    viz = visdom.Visdom(env=args.env)

    p = CFQ(_cfd=None, _dtst=True, _dset=True, _cnne=True, _vaee=None,
            _rbm=None, _cnnd=True, _vaed=None, _warp=True, _rslt=True)
#
    p.for_cfd()

    p.for_dtst(data_type=args.data_type)  # preparing the dataset
    # p.for_dset(data_path=args.train_root)  # loading the dateset
    train_data = dset.DSET(root=args.train_root)  # loading the train_dateset
    test_data = dset.DSET(root=args.test_root)  # loading the test_dateset
    train_indices, val_indices, train_loader, val_loader, test_loader, splits = p.for_dset(train_data, test_data)

    print('>>>> employing the encode-decode network...\n')
    endecoder = endecs.ConvDeconvEstimator(input_channels=args.seq_len,
                                           upsample_mode=args.upsample,
                                           )

    # print('>>>> creating warping scheme {}'.format(args.warp))
    warp = warps.__dict__[args.warp]()

    photo_loss, smooth_loss, div_loss, magn_loss, optimizer = p.for_loss()

    x, ys = torch.Tensor(), torch.Tensor()
    viz_wins = {}

    for epoch in range(1, args.epochs + 1):

        results = {}

################################################################
        for split, dl in splits.items():

            meters = AverageMeters()

            if split == 'train':
                endecoder.train(), warp.train()
            else:
                endecoder.eval(), warp.eval()

            for i, (input, targets) in enumerate(dl):

                x = input.clone().detach()
                ys = targets.clone().detach()
                ys = ys.transpose(0, 1).unsqueeze(2)

                pl, sl, dl, ml = 0, 0, 0, 0
                ims, ws = [], []

                last_im = x[:, -1, :, :].unsqueeze(1)

                for y in ys:

                    w = endecoder(x)

                    im = warp(x[:, -1, :, :].unsqueeze(1), w)
                    # print('im_size_warp(last_im_and endecoder(x)',im.size())

                    # print('x_new_before_cat_and_im', x.size())
                    x = torch.cat((x[:, 1:, :, :], im), 1)
                    # print('x_new_after_cat_and_im', x.size())

                    curr_pl = photo_loss(im, y)
                    pl += torch.mean(curr_pl)
                    sl += smooth_loss(w)
                    dl += div_loss(w)
                    ml += magn_loss(w)

                    ims.append(im.data.numpy())  # ims.append(im.cpu().data.numpy())
                    ws.append(w.data.numpy())  # ws.append(w.cpu().data.numpy())

                pl /= args.target_seq_len
                sl /= args.target_seq_len
                dl /= args.target_seq_len
                ml /= args.target_seq_len

                loss = pl + args.smooth_coef * sl + args.div_coef * dl + args.magn_coef * ml

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                meters.update(
                    dict(loss=loss.data, pl=pl.data, dl=dl.data, sl=sl.data, ml=ml.data,
                         ), n=x.size(0))

            if not args.no_plot:
                images = [
                    ('target', {'in': input.transpose(0, 1).numpy(), 'out': ys.data.numpy()}),
                    ('ws', {'out': ws}),
                    ('im', {'out': ims}), ]

                plt = plot.from_matplotlib(plot.plot_images(images))
                viz.image(plt.transpose(2, 0, 1),
                          opts=dict(title='{}, epoch {}'.format(split.upper(), epoch)),
                          win=list(splits).index(split), )

            results[split] = meters.avgs()
            print('\n\nEpoch: {} {}: {}\t'.format(epoch, split, meters))

        # transposing the results dict
        res = {}
        legend = []

        for split in results:
            legend.append(split)
            for metric, avg in results[split].items():
                res.setdefault(metric, [])
                res[metric].append(avg)

        # plotting
        for metric in res:
            y = np.expand_dims(np.array(res[metric]), 0)
            x = np.array([[epoch]*len(results)])

            if epoch == 1:
                win = viz.line(X=x, Y=y, opts=dict(showlegend=True, legend=legend, title=metric))
                viz_wins[metric] = win

            else:
                viz.line(X=x, Y=y, opts=dict(showlegend=True, legend=legend, title=metric),
                         win=viz_wins[metric],
                         update='append')

