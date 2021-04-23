class A(object):
    def __init__(self):
        self.a = self.f()

    def f(self):
        return 10


class B(A):
    def __init__(self):
        super(B, self).__init__()

    def f(self):
        return 6


a = A()
b = B()
print(a.a)
print(b.a)
