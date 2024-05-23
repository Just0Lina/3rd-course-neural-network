import numpy as np

from lab3.common import load_weights


class GRULayer:
    def __init__(self, layer_conf, weights_filename=None):
        self.layers = []
        self.input_units = layer_conf[0]["units"]
        self.hidden_units = layer_conf[1]["hidden"]
        self.output_units = layer_conf[1]["output"]
        if weights_filename:
            weights = load_weights(weights_filename)
        if weights_filename:
            self.Wr = weights.Wr
            self.Ur = weights.Ur
            self.br = weights.br

            self.Wz = weights.Wz
            self.Uz = weights.Uz
            self.bz = weights.bz

            self.Wh = weights.Wh
            self.Uh = weights.Uh
            self.bh = weights.bh

            self.W_output = weights.W_output
            self.b_output = weights.b_output
        else:
            self.initialize_weights()


    def initialize_weights(self):
        # Коэффициент для инициализации весов
        # W веса для входных данных
        # Ur веса для предыдущего скрытого состояния
        # br смещения для гейта сброса
        #
        # k - масштабирования инициализации весов
        k = 1 / np.sqrt(self.hidden_units)
        self.Wr = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Ur = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.br = np.random.uniform(-k, k, (1, self.hidden_units))

        self.Wz = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Uz = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bz = np.random.uniform(-k, k, (1, self.hidden_units))

        self.Wh = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Uh = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bh = np.random.uniform(-k, k, (1, self.hidden_units))

        # Инициализация весов для выходного слоя
        self.W_output = np.random.uniform(-k, k, (self.hidden_units, self.output_units))
        self.b_output = np.random.uniform(-k, k, (1, self.output_units))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # https://miro.medium.com/v2/resize:fit:640/format:webp/1*i-yqUwAYTo2Mz-P1Ql6MbA.png
    def forward(self, x_seq, h_prev):
        outputs=[]
        zs=[]
        rs=[]
        h_tildes=[]
        hiddens=[]
        # Проход по временным шагам в последовательности
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            # Вычисление гейтов Update Gate, z, Reset Gate, r
            z = self.sigmoid(np.dot(x_t, self.Wz) + np.dot(h_prev, self.Uz) + self.bz)
            r = self.sigmoid(np.dot(x_t, self.Wr) + np.dot(h_prev, self.Ur) + self.br)
            # Вычисление "предполагаемого" нового состояния
            h_tilde = np.tanh(np.dot(x_t, self.Wh) + np.dot((r * h_prev), self.Uh) + self.bh)
            # Обновление скрытого состояния
            hiddens.append(h_prev)
            h_prev = (1 - z) * h_prev + z * h_tilde
            outputs.append((self.softmax(np.dot(h_prev, self.W_output) + self.b_output)))
            zs.append(z)
            rs.append(r)
            h_tildes.append(h_prev)
        return hiddens, h_tildes, zs, rs, [outputs]

    def backward(self, x_seq, grad_output, hiddens , h_tildes, zs, rs, lr):
        dW_output = np.zeros_like(self.W_output)
        db_output = np.zeros_like(self.b_output)
        dWh = np.zeros_like(self.Wh)
        dUh = np.zeros_like(self.Uh)
        dbh = np.zeros_like(self.bh)
        dWz = np.zeros_like(self.Wz)
        dUz = np.zeros_like(self.Uz)
        dbz = np.zeros_like(self.bz)
        dWr = np.zeros_like(self.Wr)
        dUr = np.zeros_like(self.Ur)
        dbr = np.zeros_like(self.br)
        dh_next = np.zeros_like(hiddens[0])

        for t in reversed(range(len(x_seq))):
            x_t = x_seq[t]
            h_prev = hiddens[t]
            h_tilde = h_tildes[t]
            z = zs[t]
            r = rs[t]


            # Gradient of output layer
            dW_output += np.outer(h_tilde, grad_output[t])
            db_output += grad_output[t]

            # Gradient of hidden state to hidden state weights
            dh_output = np.dot(self.W_output, grad_output[t])
            dh = dh_output + dh_next

            # Gradient of hidden state
            dhtilde = dh * z
            dh_prev = dh * (1 - z)
            dh_tanh = dhtilde * (1 - h_tilde ** 2)
            dz = (dh * (h_tilde - h_prev)) * z * (1 - z)
            dr = (dh * (h_tilde - h_prev)) * (h_prev * r) * (1 - r)

            # Gradients for update gate (z)
            dWz += np.outer(x_t, dz)
            dUz += np.outer(h_prev, dz)
            dbz += dz

            # Gradients for reset gate (r)
            dWr += np.outer(x_t, dr)
            dUr += np.outer(h_prev, dr)
            dbr += dr

            # Gradients for hidden state to hidden state weights
            dWh += np.outer(x_t, dh_tanh)
            dUh += np.outer(r * h_prev, dh_tanh)
            dbh += dh_tanh

            dh_next = np.dot(dh_tanh, self.Uh.T)

        # Update weights using gradients
        self.W_output -= lr * dW_output
        self.b_output -= lr * db_output
        self.Wh -= lr * dWh
        self.Uh -= lr * dUh
        self.bh -= lr * dbh
        self.Wz -= lr * dWz
        self.Uz -= lr * dUz
        self.bz -= lr * dbz
        self.Wr -= lr * dWr
        self.Ur -= lr * dUr
        self.br -= lr * dbr

        return dh_prev

