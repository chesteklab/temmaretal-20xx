import torch.nn as nn
import torch.nn.functional as F
import torch
import config

device = config.device

def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.do1(self.fc1(x))))
        x = F.relu(self.bn3(self.do2(self.fc2(x))))
        x = F.relu(self.bn4(self.do3(self.fc3(x))))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores

# WillseyNet with no batchnorm or dropout
class TCNNoReg(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        scores = self.fc4(x)
        return scores

class TCNNoBN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        scores = self.fc4(x)

        return scores

class TCNNoDP(nn.Module):
    
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores
    
class RecurrentModel(nn.Module):
    ''' A general recurrent model that can use VanillaRNN/GRU/LSTM, with a linear layer to the output '''
    def __init__(self, input_size, hidden_size, num_outputs, num_layers, rnn_type='lstm', drop_prob=0,
                 hidden_noise_std=None, dropout_input=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        rnn_type = rnn_type.lower()
        self.rnn_type = rnn_type
        self.hidden_noise_std = hidden_noise_std

        if dropout_input:
            self.dropout_input = nn.Dropout(dropout_input)
        else:
            self.dropout_input = None

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True, nonlinearity='relu')
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x, h=None,  return_all_tsteps=False):
        """
        x:                  Neural data tensor of shape (batch_size, num_inputs, sequence_length)
        h:                  Hidden state tensor
        return_all_steps:   If true, returns predictions from all timesteps in the sequence. If false, only returns the
                            last step in the sequence.
        """
        x = x.permute(0, 2, 1)  # put in format (batches, sequence length (history), features)

        if self.dropout_input and self.training:
            x = self.dropout_input(x)

        if h is None:
            h = self.init_hidden(x.shape[0])

        out, h = self.rnn(x, h)
        # out shape:    (batch_size, seq_len, hidden_size) like (64, 20, 350)
        # h shape:      (n_layers, batch_size, hidden_size) like (2, 64, 350)

        if return_all_tsteps:
            out = self.fc(out)  # out now has shape (batch_size, seq_len, num_outs) like (64, 20, 2)
        else:
            out = self.fc(out[:, -1])  # out now has shape (batch_size, num_outs) like (64, 2)
        return out, h

    def init_hidden(self, batch_size):
        if self.rnn_type == 'lstm':
            # lstm - create a tuple of two hidden states
            if self.hidden_noise_std:
                hidden = (torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std).to(device=device),
                          torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std).to(device=device))
            else:
                hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device),
                          torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device))
        else:
            # not an lstm - just a single hidden state vector
            if self.hidden_noise_std:
                hidden = torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                      std=self.hidden_noise_std).to(device=device)
            else:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device)
        return hidden

class tcFNN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.do1(self.fc1(x))))
        x = F.relu(self.bn3(self.do2(self.fc2(x))))
        x = F.relu(self.bn4(self.do3(self.fc3(x))))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores

# WillseyNet with no batchnorm or dropout
class noreg_tcFNN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        scores = self.fc4(x)
        return scores

class tcFNN_nobn(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        scores = self.fc4(x)

        return scores

class tcFNN_nodp(nn.Module):
    
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores