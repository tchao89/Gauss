import numpy as np
import shelve

print(np.exp([1, 2, 3]))
print(np.power([1, 2, 3], 0.5))
params = ()

# le = {"switch": {"func": np.power, "params": (0.5, )}}
# with shelve.open("./label_encodings") as shelve_open:
#     shelve_open['label_encoding'] = le

with shelve.open("./label_encodings") as shelve_open:
    le = shelve_open['label_encoding']
print(le["switch"]["func"]([1, 2, 3], le["switch"]["params"]))
