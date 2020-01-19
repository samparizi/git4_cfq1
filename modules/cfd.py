

class CFD:

    def __init__(self, data=1):
        super(CFD, self).__init__()
        self.data = data

    def test_cfd(self, data):
        return data + 1


if __name__ == "__main__":
    p = CFD(data=1)
