import numpy as np
import shelve

print(np.exp([1, 2, 3]))
print(np.power([1, 2, 3], 0.5))
params = ()

# le = {"switch": {"func": np.power, "params": (0.5, )}}
# with shelve.open("./label_encodings") as shelve_open:
#     shelve_open['label_encoding'] = le

# with shelve.open("./label_encodings") as shelve_open:
#     le = shelve_open['label_encoding']
# print(le["switch"]["func"]([1, 2, 3], le["switch"]["params"]))
print(np.iinfo(np.int8).min)
print(np.iinfo(np.int8).max)
a = np.array([127.1, -128.0, 0, 1], dtype=np.int8)
print(a)
