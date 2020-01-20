import visdom
import torchvision.transforms as transforms


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

    def for_cfd(self, data):

        if self._cfd is None:
            print('cfd is NOT active')
            pass
        else:
            print('cfd is active')
            return cfd.CFD.test_cfd(self, data) + 1

    def for_dtst(self, data_type):

        if self._dtst is None:
            print('dtst is NOT active')
            pass
        else:
            print('dtst is active')
            return dtst.DTST.test_dtst(self, data_type)

    def for_dset(self, data_path):

        if self._dset is None:
            print('dset is NOT active')
            pass
        else:
            print('dset is active')
            data = dset.DSET.test_dset(self)
            # dset = DSET(data_path)
            # dset.DSET.test_dset(data)
            # CFQ.dset.DSET.

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
    # viz = visdom.Visdom(env=args.env)

    p = CFQ(_cfd=None, _dtst=True, _dset=True, _cnne=True, _vaee=None,
            _rbm=None, _cnnd=True, _vaed=None, _warp=True, _rslt=True)
#
    p.for_cfd(data=1)
    p.for_dtst(data_type=args.data_type)  # preparing the dataset
    p.for_dset(data_path=args.train_root)  # loading the dateset
    dset = dset.DSET(args.train_root)   # loading the dateset
    # test_dset = p.DSET(args.test_root, seq_len=args.seq_len,
    #                       target_seq_len=args.target_seq_len,
    #                       transform=transforms.Compose([transforms.ToTensor()]),
    #                       )

#     # p = p.for_rslt(data=1)
#     print('CFQobject', p)
