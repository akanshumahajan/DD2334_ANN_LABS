# %%
import numpy as np
from matplotlib import pyplot as plt

from generate_data import generate_data
plt.style.use('ggplot')


class NeuralNetwork():
    def __init__(self, method='batch', alpha=0.5,learning_rate=0.00001, n_inputs=2, n_hidden=5, n_outputs=1):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.method = method
        self.learning_rate = learning_rate
        self.alpha = alpha
        self._init_network()

    def _init_network(self):
        self.x = np.zeros(self.n_inputs)
        self.h = np.zeros(self.n_hidden)
        self.y = np.zeros(self.n_outputs)

        cov_v = np.eye(self.n_inputs+1)
        mean_v = np.zeros(self.n_inputs+1)
        self.v = np.random.multivariate_normal(
            mean_v, cov_v, self.n_hidden)
        self.v_momentum = np.zeros((self.n_hidden, self.n_inputs+1))

        cov_w = np.eye(self.n_hidden+1)
        mean_w = np.zeros(self.n_hidden+1)
        self.w = np.random.multivariate_normal(
            mean_w, cov_w, self.n_outputs)
        self.w_momentum = np.zeros((self.n_outputs, self.n_hidden+1))


    def _add_bias(self):
        self.data = np.column_stack((self.data, np.ones(self.data.shape[0])))

    def _activation(self, x):
        return np.tanh(x)

    def _activation_deriv(self, x):
        return 1-np.tanh(x)**2

    def _forward_pass(self):
        self.h = self._activation(np.dot(self.v, self.data.T))
        self.h = np.row_stack((self.h, np.ones(self.n_data)))  # add bias
        self.y = self._activation(np.dot(self.w, self.h))

    def _backward_pass(self, labels):
        # calc delta_k
        y_in = np.dot(self.w, self.h)
        self.delta_k = (labels - self.y) * self._activation_deriv(y_in)

        # calc delta_j
        h_in = np.dot(self.v, self.data.T)
        h_in = np.row_stack((h_in, np.ones(self.n_data)))  # add bias
        delta_k_w = np.dot(np.array(self.delta_k).T, self.w)
        h_in_deriv = self._activation_deriv((h_in))
        self.delta_j = np.multiply(delta_k_w.T, h_in_deriv)
        # print(h_in_deriv.shape, self.delta_j.shape)

    def _weight_updating(self, index):
        self.w_momentum = self.alpha * self.w_momentum - \
            (1-self.alpha)*self.delta_k[:, index] * self.h[:, index]
        self.w = self.w + self.learning_rate * self.w_momentum
        
        self.v_momentum = self.alpha * self.v_momentum - (1-self.alpha)*np.dot(np.array([self.delta_j[0:-1, index]]).T,
                                                                               np.array([self.data[index, :]]))
        self.v = self.v + self.learning_rate * self.v_momentum
            

    def fit(self, data, labels, n_epochs):
        '''
        Expects data in the form
        data = [[x1,y1],
                [x2,y2],
                [x3,y3],
                ...
                [xN,yN]]
        '''
        self.labels = labels
        self.n_data = len(labels)
        self.data = data
        self.mse = np.zeros(n_epochs+1)
        self.miss_ratio = np.zeros(n_epochs+1)
        self.misses = np.zeros(n_epochs+1)
        self._add_bias()

        for i in range(n_epochs):
            if self.method == 'sequential':
                for index in range(self.n_data):
                    self._forward_pass()
                    self._backward_pass(labels)
                    self._weight_updating(index)

            if self.method == 'batch':
                self._forward_pass()
                self._backward_pass(labels)
                for index in range(self.n_data):
                    self._weight_updating(index)
            self.result = list(map(classifier, self.y[0, :]))
            self.mse[i] = np.sum(
                np.square(self.labels-self.y[0, :]))/self.n_data
            self.misses[i] = self.n_data - \
                np.count_nonzero(self.result == self.labels)

    def predict(self, data_point):
        data_point = np.concatenate((data_point, np.ones(1)))
        h = self._activation(np.dot(self.v, data_point))
        h = np.concatenate((h, np.ones(1)))  # add bias
        y = self._activation(np.dot(self.w, h))
        return y

    def metrics(self):
        self._forward_pass()
        self.mse[-1] = np.sum(
            np.square(self.labels-self.y[0, :]))/self.n_data
        self.misses[-1] = self.n_data - \
            np.count_nonzero(self.result == self.labels)
        return self.mse, self.misses

        # for i in range(self.n_data):
        #     print(labels[i] == self.result[i])


def classifier(val):
    if val >= 0:
        return 1
    else:
        return -1


def cut_data(data, cut_a, cut_b):
    class_a = data[data[:, 2] == 1]
    class_b = data[data[:, 2] == -1]
    np.random.shuffle(class_a)
    np.random.shuffle(class_b)
    n_a = int(len(class_a)*(1-cut_a))
    n_b = int(len(class_b)*(1-cut_b))

    train_a = class_a[:n_a]
    train_b = class_b[:n_b]
    valid_a = class_a[n_a:]
    valid_b = class_b[n_b:]

    train_set = np.row_stack([train_a, train_b])
    np.random.shuffle(train_set)
    valid_set = np.row_stack([valid_a, valid_b])
    np.random.shuffle(valid_set)

    return train_set, valid_set


def cut_asymmetric(data):
    class_a = data[data[:, 2] == 1]
    class_b = data[data[:, 2] == -1]

    class_a_upper = class_a[class_a[:, 0] > 0]
    class_a_lower = class_a[class_a[:, 0] < 0]

    n_a_lower = int(len(class_a_lower)*0.2)
    n_a_upper = int(len(class_a_upper)*0.8)

    class_a_upper_train = class_a_upper[:n_a_upper]
    class_a_upper_valid = class_a_upper[n_a_upper:]
    class_a_lower_train = class_a_lower[:n_a_lower]
    class_a_lower_valid = class_a_lower[n_a_lower:]
    valid_set = np.row_stack([class_a_upper_valid, class_a_lower_valid])

    class_a_train = np.row_stack([class_a_upper_train, class_a_lower_train])
    train_set = np.row_stack([class_a_train, class_b])

    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)

    return train_set, valid_set


def plot_decision_boundry(data, res, predictor):
    x_min = np.min(data[:, 0])-0.5
    x_max = np.max(data[:, 0])+0.5
    y_min = np.min(data[:, 1])-0.5
    y_max = np.max(data[:, 1])+0.5

    x_out = np.linspace(x_min, x_max, res)
    y_range = np.linspace(y_min, y_max, res)
    result = np.ones((res, res))
    y_out = np.zeros(res)
    for row in range(res):
        for col in range(res):
            result[row, col] = np.abs(predictor([x_out[col], y_range[row]]))

    for i in range(res):
        y_out[i] = y_range[np.argmin(result[:, i])]

    plt.plot(x_out, y_out)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Decision Boundry, #hidden nodes=5')


def test_num_nodes():
    test = [1,5,10,15,25]#[1, 5, 10, 15, 20, 30]
    N = 100

    
    epochs = np.arange(101)
    mse = np.zeros(101)
    miss = np.zeros(101)
    for i in test:
        data = generate_data(N, plot=False, meanA=[2, 2.5], meanB=[
            0, 0], sigmaA=0.8, sigmaB=0.5)
        for j in range(200):
            network = NeuralNetwork(method='batch', n_inputs=2,
                                    n_hidden=i, n_outputs=1)
            network.fit(data[:, 0:2], data[:, 2], n_epochs=100)
            temp_mse, temp_miss = network.metrics()
            mse += temp_mse
            miss += temp_miss
        mse = mse/200
        miss = miss/200
        plt.subplot(1,2,1)
        plt.plot(epochs, mse, label='# hidden nodes='+str(i))
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title("Mean squared error")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, miss, label='# hidden nodes='+str(i))
        plt.xlabel('Epochs')
        plt.ylabel('# of missclassifications')
        plt.title("Missclassifications")
        plt.legend()
        
    
    plt.show()
def test_network():
    N = 100
    network = NeuralNetwork(method='batch', n_inputs=2,
                            n_hidden=5, n_outputs=1)
                            # n_hidden

    # data = generate_data(N, plot=True, meanA=[0, 0], meanB=[
    #     10, 10], sigmaA=2, sigmaB=2)
    # data = generate_data(N, plot=True, meanA=[2, 1.5], meanB=[
    # 0, -1], sigmaA=0.2, sigmaB=0.3)

    np.random.seed(2)
    data = generate_data(N, plot=True, meanA=[2, 2.5], meanB=[
        0, 0], sigmaA=0.8, sigmaB=0.5)
    
    network.fit(data[:, 0:2], data[:, 2], n_epochs=100)
    mse, miss_ratio = network.metrics()
    print(mse[-1], miss_ratio[-1])
    plot_decision_boundry(data, 1000, network.predict)
    plt.show()

def test_cut_data():
    pass
test_network()
# test_num_nodes() # choose 5 nodes
# test_cut_data()
