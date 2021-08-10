import ctypes

lib = ctypes.cdll.LoadLibrary("/home/liangqian/PycharmProjects/Gauss/test_dataset/TestLib.so")

print("Test int io...")
func = lib.receiveInt
func.argtypes = [ctypes.c_int]
func.restype = ctypes.c_int
print(func(100))
print()

print("Test double io...")
func = lib.receiveDouble
func.argtypes = [ctypes.c_double]
func.restype = ctypes.c_double
print(func(3.14))
print()

print("Test char io...")
func = lib.receiveChar
func.argtypes = [ctypes.c_char]
func.restype = ctypes.c_char
print(func(ctypes.c_char(b'a')))
print()

print("Test string io...")
func = lib.receiveString
func.argtypes = [ctypes.c_char_p]
func.restype = ctypes.c_char_p
print(func(b"(This is a test string.)"))
print()

print("Test struct io...")


class Struct(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p),
                ('age', ctypes.c_int),
                ('score', ctypes.c_float * 3)]


lib.argtypes = [Struct]
lib.receiveStruct.restype = Struct
array = [85, 93, 95]
st = lib.receiveStruct(Struct(b'XiaoMing', 16, (ctypes.c_float * 3)(*array)))
print(str(st.name) + ' ' + str(st.age) + ' ' + str(st.score[0]) + ' in python')
print()

print('Test struct pointer io...')
lib.receiveStructPtr.restype = ctypes.POINTER(Struct)
lib.receiveStructPtr.argtypes = [ctypes.POINTER(Struct)]
p = lib.receiveStructPtr(Struct(b"XiaoHuang", 19, (ctypes.c_float * 3)(*array)))
print(str(p.contents.name) + ' ' + str(p.contents.age) + ' ' + str(p.contents.score[0]) + ' in python')
print()

print('Test struct array io...')
lib.receiveStructArray.restype = ctypes.POINTER(Struct)
lib.receiveStructArray.argtypes = [ctypes.ARRAY(Struct, 2), ctypes.c_int]
array = [Struct(b'student1', 19, (ctypes.c_float * 3)(91, 92, 93)),
         Struct(b'student2', 18, (ctypes.c_float * 3)(88, 95, 92))]
p = lib.receiveStructArray(ctypes.ARRAY(Struct, 2)(*array), 2)
print(str(p.contents.name) + ' ' + str(p.contents.age) + ' ' + str(p.contents.score[2]) + ' in python')
