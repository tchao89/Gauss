import abc

class A(metaclass=abc.ABCMeta):
 @abc.abstractmethod
 def run(self):
  pass
class B(A):
 @abc.abstractmethod
 def test(self):
   pass
class C(B):
 def run(self):
   print("sb")
 def test(self):
  print("fuck")
tt = C()
tt.run()
tt.test()
