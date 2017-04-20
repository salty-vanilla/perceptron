import numpy as np


class Perceptron():
    def __init__(self, dim_hidden):
        self.weight = np.random.randn(dim_hidden)
        self.bias = np.random.randn(1)

    def predict(self, x):
        return self.step_function(np.dot(x, self.weight) + self.bias)

    def fit(self, x, y, nb_epochs=1, lr=0.01, shuffle=True, verbose=1):
        assert len(x) == len(y)

        indexes = np.arange(len(x))
        if shuffle:
            indexes = np.random.permutation(indexes)

        for epoch in range(nb_epochs):
            correct = 0
            if verbose == 1:
                print("\nepoch : {} / {}".format(epoch + 1, nb_epochs))
            for iter_, index in enumerate(indexes):
                correct += self.update(x[index], y[index], lr)
                accuracy = float(correct / (iter_ + 1))
                if verbose == 1:
                    print("iter : {:} / {:}  acc : {:.3f}".
                          format(index + 1, len(x), accuracy), end='\r')
        print("\nTraining is done ...")

    def update(self, x, y_true, lr):
        y_pred = self.predict(x)

        if y_pred != y_true:
            self.weight += lr * x * y_true
            self.bias += lr * 1. * y_true
            return 0
        else:
            return 1

    def eval(self, x, y):
        assert len(x) == len(y)
        correct = 0
        for i in range(len(x)):
            y_pred = self.predict(x[i])
            y_true = y[i]

            if y_pred == y_true:
                correct += 1
        return float(correct / len(x))

    @staticmethod
    def step_function(x):
        if x < 0:
            return -1
        else:
            return +1


def data_init(split=0.2):
    def load_from_dat(file_path):
        return np.loadtxt(file_path)

    def shuffle(x, y):
        assert len(x) == len(y)
        indexes = np.arange(len(x))
        indexes = np.random.permutation(indexes)
        return x[indexes], y[indexes]

    data_0 = load_from_dat('data0.dat')
    data_1 = load_from_dat('data1.dat')
    x = np.append(data_0, data_1, axis=0)
    y = np.array([-1] * len(data_0) + [1] * len(data_1))

    x, y = shuffle(x, y)

    train_x, train_y = x[:int(len(x) * (1 - split))], y[:int(len(y) * (1 - split))]
    test_x, test_y = x[len(train_x):], y[len(train_x):]

    return (train_x, train_y), (test_x, test_y)


def main():
    for split in np.arange(0.1, 1., 0.1):
        (train_x, train_y), (test_x, test_y) = data_init(split=split)
        dim = train_x[0].shape[0]
        percptron = Perceptron(dim_hidden=dim)
        percptron.fit(train_x, train_y, nb_epochs=20, verbose=0)
        train_accuracy = percptron.eval(train_x, train_y)
        test_accuracy = percptron.eval(test_x, test_y)
        print("* Split : {:.1f} Train Accuracy : {:.3f} Test Accuracy: {:.3f}".
              format(split, train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
