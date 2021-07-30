class A:
    def __init__(self, a, b, c):
        self.value = a
        self.b = b
        self.c = c

class lgb_model(object):
    def __init__(self, x, y="test model", z=None):
        if z is None:
            z = {"model": "a"}

        self.a = A(x, y, z)

class ModelRepr(object):

    def __init__(self):
        self._model = None
        self._best_model = None

    def train(self, x):
        self._model = lgb_model(x)

    def eval(self):
        return self._model.x

    def update_best_model(self):
        assert self._model is not None

        if self._best_model is None:

            self._best_model = self._model
        else:

            if self._model.a.value > self._best_model.a.value:
                self._best_model = self._model

    @property
    def best_model(self):
        return self._best_model

    @property
    def last_model(self):
        return self._model


data = [6, 5, 3, 7, 12, 9, 10]
model_repr = ModelRepr()

for i in data:
    model_repr.train(x=i)
    model_repr.update_best_model()
    print("best model: ", model_repr.best_model.a.value)
    print("last model: ", model_repr.last_model.a.value)
