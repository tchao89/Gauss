def fun1(fun):
    print(1)
    t = fun(10)
    print(t)


def fun2(x):
    print(2)
    return x


fun1(fun2)
