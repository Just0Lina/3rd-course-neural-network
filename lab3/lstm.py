import numpy as np

from lab3.common import load_weights


class LSTMLayer:
    def __init__(self, layer_conf, weights_filename=None):
        self.layers = []
        self.input_units = layer_conf[0]["units"]
        self.hidden_units = layer_conf[1]["hidden"]
        self.output_units = layer_conf[1]["output"]
        if weights_filename:
            weights = load_weights(weights_filename)
        if weights_filename:
            self.Wf = weights.Wf
            self.Uf = weights.Uf
            self.bf = weights.bf

            self.Wi = weights.Wi
            self.Ui = weights.Ui
            self.bi = weights.bi

            self.Wc = weights.Wc
            self.Uc = weights.Uc
            self.bc = weights.bc

            self.Wo = weights.Wo
            self.Uo = weights.Uo
            self.bo = weights.bo

            self.W_output = weights.W_output
            self.b_output = weights.b_output
        else:
            self.initialize_weights()

    def initialize_weights(self):
        # Коэффициент для инициализации весов
        k = 1 / np.sqrt(self.hidden_units)

        # Веса для форгет гейта
        self.Wf = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Uf = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bf = np.random.uniform(-k, k, (1, self.hidden_units))

        # Веса для input гейта
        self.Wi = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Ui = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bi = np.random.uniform(-k, k, (1, self.hidden_units))

        # Веса для cell state
        self.Wc = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Uc = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bc = np.random.uniform(-k, k, (1, self.hidden_units))

        # Веса для output гейта
        self.Wo = np.random.uniform(-k, k, (self.input_units, self.hidden_units))
        self.Uo = np.random.uniform(-k, k, (self.hidden_units, self.hidden_units))
        self.bo = np.random.uniform(-k, k, (1, self.hidden_units))

        # Инициализация весов для выходного слоя
        self.W_output = np.random.uniform(-k, k, (self.hidden_units, self.output_units))
        self.b_output = np.random.uniform(-k, k, (1, self.output_units))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x_seq, h_prev, c_prev):
        outputs = []
        fs = []
        is_ = []
        cs_ = []
        os = []
        hiddens_outputs=[]
        hiddens = []
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]

            # Forget gate
            f = self.sigmoid(np.dot(x_t, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
            fs.append(f)

            # Input gate
            i = self.sigmoid(np.dot(x_t, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
            is_.append(i)

            # Candidate cell state
            c_ = self.tanh(np.dot(x_t, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
            cs_.append(c_)

            # Update cell state
            c = f * c_prev + i * c_
            c_prev = c

            # Output gate
            o = self.sigmoid(np.dot(x_t, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
            os.append(o)

            # Hidden state
            h = o * self.tanh(c)
            hiddens.append(h)

            # Output layer
            output = self.softmax(np.dot(h, self.W_output) + self.b_output)
            outputs.append(output)

            h_prev = h

        return hiddens, c_prev[0], [outputs], fs, is_, cs_, os[0][0]

    def backward(self, x_seq, grad, hiddens, c_prev, fs, is_, cs_, os, lr):
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUc = np.zeros_like(self.Uc)
        dUo = np.zeros_like(self.Uo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dc_prev = np.zeros_like(c_prev)
        dh_next = np.zeros_like(hiddens[0])

        for t in reversed(range(len(x_seq)-1)):
            dh = grad[t] + dh_next

            # Output gate
            do = (dh * self.tanh(c_prev[t]) * self.sigmoid(os[t]) * (1 - self.sigmoid(os[t])))[0]
            dUo += np.outer(hiddens[t], do)
            dWo += np.outer(x_seq[t], do)
            dbo += do
            dh_next = np.dot(do, self.Uo)

            # Hidden state
            dc = (dh * os[t] * (1 - self.tanh(c_prev[t]) ** 2) + dc_prev)[0]
            dUc += np.outer(hiddens[t], dc)
            dWc += np.outer(x_seq[t], dc)
            dbc += dc
            dh_ = np.dot(dc, self.Uc)

            # Candidate cell state
            dc_ = dc * is_[t]
            dUi += np.outer(hiddens[t], dc_)
            dWc += np.outer(x_seq[t], dc_)
            dbi += dc_[0]
            dh_ += np.dot(dc_, self.Ui[0])

            # Input gate
            di = dc * cs_[t]
            dUi += np.outer(hiddens[t], di)
            dWi += np.outer(x_seq[t], di)
            dbi += di[0]
            dh_ += np.dot(di, self.Ui[0])

            # Forget gate
            df = dc * c_prev[t - 1]
            dUf += np.outer(hiddens[t], df)
            dWf += np.outer(x_seq[t], df)
            dbf += df[0]
            dh_ += np.dot(df, self.Uf[0])

            # Update cell state
            dc_prev = dc * fs[t]

        self.Wf -= lr * dWf
        self.Wi -= lr * dWi
        self.Wc -= lr * dWc
        self.Wo -= lr * dWo
        self.Uf -= lr * dUf
        self.Ui -= lr * dUi
        self.Uc -= lr * dUc
        self.Uo -= lr * dUo
        self.bf -= lr * dbf
        self.bi -= lr * dbi
        self.bc -= lr * dbc
        self.bo -= lr * dbo

