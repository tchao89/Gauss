import itertools

a = [1, 2]
b = [3, 4]
c = [5, 6]
d = [7, 8]
c = itertools.product(a, b, c, d)
for i in c:
    print(i)
