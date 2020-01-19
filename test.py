class First(object):
    def __init__(self):
        super(First, self).__init__()
        print("first")


class Second(object):
    def __init__(self):
        super(Second, self).__init__()
        print("second")

    def sec_s(self,  d):
        return d + 1

class Third(First, Second):
    def __init__(self, data):
        super(Third, self).__init__()
        self.data = data
        print("third")

    def th_s(self,  data):
        data = Second.sec_s(self, data) + 1
        return data + 1


class A(object):
    def __init__(self,a):
        self.a=a


class B(A):
    def __init__(self,b,**kw):
        self.b=b
        super(B,self).__init__(**kw)


class C(A):
    def __init__(self,c,**kw):
        self.c=c
        super(C,self).__init__(**kw)


class D(B,C):
    def __init__(self,a,b,c,d):
        super(D,self).__init__(a=a,b=b,c=c)
        self.d=d


if __name__ == "__main__":
    p = Third(data=1)
    print('p', p)
    p = p.th_s(data=2)
    print('p', p)

    s = Second()
    s = s.sec_s(d=20)
    print('s', s)

    d = D(a=1, b=2, c=3, d=4)
