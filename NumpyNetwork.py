import numpy as np 

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)  # weigths for forget gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)  # weigths for input gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)  # weights for output gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)  # weights for memory 

        self.bf = np.zeros((hidden_size, 1))  # bias forget 
        self.bi = np.zeros((hidden_size, 1))  # bias input
        self.bo = np.zeros((hidden_size, 1))  # bias output
        self.bc = np.zeros((hidden_size, 1))  # bias memory

    def forward(self, x, prev_h, prev_c):
        input_concat = np.vstack((prev_h, x))
        
        forget_gate = sigmoid(np.dot(self.Wf, input_concat) + self.bf)
        input_gate = sigmoid(np.dot(self.Wi, input_concat) + self.bi)
        output_gate = sigmoid(np.dot(self.Wo, input_concat) + self.bo)
        cell_state = np.tanh(np.dot(self.Wc, input_concat) + self.bc)
        
        c = forget_gate * prev_c + input_gate * cell_state
        h = output_gate * np.tanh(c)
        
        return h, c

#  LSTM network
class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.W_output = np.random.randn(output_size, hidden_size)  # Веса для выходного слоя
        self.b_output = np.zeros((output_size, 1))  # Смещение для выходного слоя

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        outputs = []
        for x in inputs:
            h, c = self.lstm_cell.forward(x.reshape(-1, 1), h, c)
            output = np.dot(self.W_output, h) + self.b_output
            outputs.append(output)

        return outputs
