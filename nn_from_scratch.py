import numpy as np


def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def sigmoid_gradient(z):
    return(sigmoid(z)*(1-sigmoid(z)))

def get_cost_value(y_best, y_true):
    cost = - np.mean(np.dot(np.log(y_best).T, y_true) +
            np.dot(np.log(1 - y_best).T, 1 - y_true))
    return cost

class Layer:
    '''
    Every input (x) is a matrix (m x n) with m->n_samples and n->n_features
    Given N->n_neurons, the weight matrix (W) is defined as (n x N), so that:
    dim(x dot W) = dim(z)
    (m x n) x (n x N) = (m x N)
    '''
    def __init__(self, *args):
        self.W = None
        self.b = None
        self.activation_function = None
        if args:
            n_features, n_neurons, activation_function = args
            self.W = np.random.rand(n_features, n_neurons)
            self.b = np.zeros((1, n_neurons))
            self.activation_function = activation_function

    def forward(self, A_prev):
        self.Z = np.dot(A_prev, self.W) + self.b
        self.A = self.activation_function(self.Z)

    def backward(self, delta, Z_prev, A_prev):
        self.delta = delta
        self.nabla_W = np.dot(A_prev.T, self.delta)
        self.nabla_b = np.sum(self.delta)
        self.delta_prev = np.dot(self.delta, self.W.T) * sigmoid_gradient(Z_prev)

    def update(self, learning_rate):
        self.W -= learning_rate * self.nabla_W
        self.b -= learning_rate * self.nabla_b

class NeuralNetwork:
    def __init__(self, n_input_features, output_tpl, hidden_layers = None):
        np.random.seed(0)
        if hidden_layers is None:
            hidden_layers = []
        layers_tpl = hidden_layers + [output_tpl]
        self.layers = []
        layers_inputs = [n_input_features] + [tpl[0] for tpl in layers_tpl[:-1]]
        for n_features, n_neurons, activation_function in zip(
                layers_inputs,
                [tpl[0] for tpl in layers_tpl],
                [tpl[1] for tpl in layers_tpl]):
            self.layers.append(Layer(n_features, n_neurons, activation_function))
        self.layers.insert(0, Layer()) # Layer 0, it has only self.a

    def forward(self, inputs):
        self.layers[0].A = inputs
        self.layers[0].Z = inputs
        for index, layer in enumerate(self.layers[1:]):
            layer.forward(self.layers[index].A)
        self.y_best = self.layers[-1].A

    def backward(self, y_true, learning_rate):
        delta = (self.layers[-1].A - y_true) * sigmoid_gradient(
                self.layers[-1].Z)
        Z_prev = self.layers[-2].Z
        A_prev = self.layers[-2].A
        for index, layer in reversed(list(enumerate(self.layers[1:]))):
            layer.backward(delta, Z_prev, A_prev)
            delta = layer.delta_prev
            Z_prev = self.layers[index - 1].Z
            A_prev = self.layers[index - 1].A
            layer.update(learning_rate)

    def train(self, inputs, y_true, epochs, learning_rate):
        self.cost_history = []
        for epoch in range(epochs):
            self.forward(inputs)
            cost = get_cost_value(self.y_best, y_true)
            self.cost_history.append(cost)
            self.backward(y_true, learning_rate)
            if epoch%100 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, cost))
