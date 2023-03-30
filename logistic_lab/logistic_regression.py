import numpy as np


def sigmoid(x, w):
    z = np.dot(x,w)

    return 1 / (1 + np.exp(z))


class LogisticRegression:
    def __init__(self, n_features):
        self._n_features = n_features
        self.w = np.random.randn(n_features) * 0.001
        self.alpha = 0.01

    def predict(self,x):

        return self._forward(x)

    def _forward(self, x, *args, **kwargs):
        #print(self.w)
        return sigmoid(x, self.w)

    def loss(self, xt, yt, H):
        # binary cross-entropy
        m = xt.shape[0]
        # y = y.reshape(y.shape[0], 1)

        #H = self._forward(xt)
        #print(H)
        return (sum(yt * np.log(H) + (1 - yt) * np.log(1 - H))) / (-m)

    def gradientDescent(self, x_t, y_t,y_pred):
        m = x_t.shape[0]
        self.w -= self.alpha/m * x_t.T.dot(y_t - y_pred)

    def fit(self,x,y,epochs=100,verbose=False):

        for e in range(epochs):
            y_pred = self._forward(x)
            l = self.loss(x,y,y_pred)
            if e % 200 == 0 and verbose:
                print(f'Epoch {e}/{epochs} -> loss: {l}')
            self.gradientDescent(x,y,y_pred)

    def test(self, x_test, y_test):
        p = np.zeros((x_test.shape[0], 1))
        for (i, example) in enumerate(x_test):
            prob = self._forward(example)
            if prob >= 0.5:
                p[i] = 1
            else:
                p[i] = 0
        print('Training Accuracy: {}%'.format(np.mean(p == y_test.reshape(p.shape)) * 100))




