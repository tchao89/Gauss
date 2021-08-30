import sys
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from entity.dataset.plain_dataset import PlaintextDataset

class CONFIGS:
    filepath_csv = "./test_dataset/bank_numerical.csv"
    filepath_txt = "./test_dataset/bank_numerical.txt"
    filepath_libsvm = "./test_dataset/bank_numerical.libsvm"
    cls = "classification"
    target_name = ["deposit"]

testset_csv = PlaintextDataset(
    name="testset",
    data_path=CONFIGS.filepath_csv,
    task_type=CONFIGS.cls,
    target_name=CONFIGS.target_name,
    memory_only=True
)
testset_val_csv = testset_csv.split()
print(testset_csv)
print(testset_val_csv)
print("+ "*30)
# testset_csv.union(testset_val_csv)
# print(testset_csv)
# print("= " * 50)

# testset_txt = PlaintextDataset(
#     name="testset",
#     data_path=CONFIGS.filepath_txt,
#     task_type=CONFIGS.cls,
#     target_name=CONFIGS.target_name,
#     memory_only=True
# )
# testset_val_txt = testset_txt.split()
# print(testset_txt.get_dataset())
# print(testset_val_txt)
# print("= " * 30)

# testset_lib = PlaintextDataset(
#     name="testset",
#     data_path=CONFIGS.filepath_libsvm,
#     task_type=CONFIGS.cls,
#     target_name=CONFIGS.target_name,
#     memory_only=True
# )
# testset_val_lib = testset_lib.split()
# print(testset_lib.get_dataset())
# print(testset_val_lib.get_dataset())