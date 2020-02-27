# %%
import numpy as np
from matplotlib import pyplot as plt

VAR = 0.1


class RBFNetwork():
    def __init__(self, n_inputs, n_rbf, n_outputs, learning_rate=0.1, min_val=0.05, max_val=2*np.pi):
        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.rbf_centers = np.array([np.linspace(min_val, max_val, n_rbf)])
        self.w = np.array([np.random.normal(0, 1, n_rbf)])
        self.RBF = np.vectorize(self._base_func)

    def _base_func(self, x, center):
        return np.exp(-np.linalg.norm(x-center)**2/(2*VAR**2))

    def fit(self, data, f, method='delta'): #Sin2 train is data and f
        self.data = np.array([data]).T
        if method == 'batch':
            phi = self.RBF(self.data, self.rbf_centers)
            self.w = np.dot(
                np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)

        if method == 'delta':
            self.learning_rate= 0.01
            print((len(data)))
            for i in range (len(data)):
                #print((data[i]))
                #print(data)
                data_bar = data[i]
                #print(data_bar)
                #print(self.rbf_centers[:,i])
                rbf_centres_bar=self.rbf_centers
                #print(rbf_centres_bar)
                #print(data_bar-rbf_centres_bar)
                phi_dash= np.exp(-np.linalg.norm(data_bar-rbf_centres_bar)**2/(2*VAR**2)) # phi xk
                #phi_dash=phi_dash.transpose()
                #print((phi_dash.shape))
                f_dash = self.w * phi_dash
                f_dash_out= np.sum(f_dash)
                #print(f_dash_out)
                #print(f)
                #print(f[i])
                f_actual= f[i]
                error = f_actual-f_dash_out
                eeta_dash = 0.5*(error)**2
                delta_w = self.learning_rate*error*phi_dash
                #print((delta_w.shape))
                self.w += delta_w
                #print("Sbsrgfgbyaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", self.w)

            # print(w)

    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T)


def plot_prediction():
    network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y[i] = network.predict(x_i)


def sin2(x):
    return np.sin(2*x)


def square(x):
    if np.sin(x) >= 0:
        return 1
    else:
        return -1


def generate_input(start):
    patterns = np.arange(start, 2*np.pi, 0.1)
    np.random.shuffle(patterns)
    return patterns


x_train = generate_input(0)
x_test = generate_input(0.05)
sin2_train = list(map(sin2, generate_input(0)))
sin2_test = list(map(sin2, generate_input(0.05)))

square_train = list(map(square, generate_input(0)))
square_test = list(map(square, generate_input(0.05)))

network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)
network.fit(sin2_train, sin2_train)
print(network.predict([0.5, 0]))
print(sin2(np.array([0.5, 0])))
# network.RBF(0.5, 0.45)
# print(network.w)
