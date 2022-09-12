import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# model 1: naive lstm
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        size= r_out.shape[0]
        out = self.out(h_n.view(size, -1))
        return out


# model 2: sparse lstm
class Sparse_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, sparse_output_dim, output_dim):
        super().__init__()
        # A sparse LSTM network consists of stacked LSTM layers that are preceded by a fully connected feedforward layer
        self.sparse_fc = nn.Linear(input_dim, sparse_output_dim)
        self.lstm = nn.LSTM(sparse_output_dim, hidden_dim, num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.sparse_fc(x)
        x = F.relu(x, inplace=False)
        out, (h_n, h_c) = self.lstm(x, None)
        out_res = self.out_fc(out)
        return out_res[:, -1, :]


# model 3: sparse encoder decoder
class Sparse_Encoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, sparse_output_dim):
        super().__init__()
        # The sparse ED network is similar to the sparse LSTM network in the sense that there is a fully connected
        # feedforward layer that precedes the ED architecture.
        self.lstm = nn.LSTM(sparse_output_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, hidden = self.lstm(x, None)
        # hidden = (h_n.view(config.batch_size, -1, config.hidden_dim),
        #           h_c.view(config.batch_size, -1, config.hidden_dim))
        return out, hidden


class Sparse_Decoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, _ = self.lstm(x, hidden)
        out = self.out_fc(out)
        return out


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, sparse_output_dim, output_dim):
        super().__init__()
        self.sparse_fc = nn.Linear(input_dim, sparse_output_dim)
        self.encoder = Sparse_Encoder(hidden_dim, num_layers, sparse_output_dim)
        self.decoder = Sparse_Decoder(hidden_dim, num_layers, output_dim)

    def forward(self, x):
        x = self.sparse_fc(x)
        x = F.relu(x, inplace=False)
        enc_out, hidden = self.encoder(x)
        dec_out = self.decoder(enc_out, hidden)
        return dec_out[:, -1, :]


# model 4: seq2seq with attention
# reference: https://wmathor.com/index.php/archives/1451/
class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        # bidirectional GRU / LSTM is used
        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.transpose(0, 1)
        enc_output, enc_hidden = self.rnn(src, None)
        # enc_output = [seq_length, batch_size, enc_hid_dim * num_directions] = [24, 64, 256]
        # enc_hidden = [n_layers * num_directions, batch_size, enc_hid_dim] = [2, 64, 128]
        # enc_hidden is stacked[forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output is always from the last layers
        # enc_hidden[-2, :, :] is the last of the forwards RNN
        # enc_hidden[-1, :, :] is the last if the backwards RNN

        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        # s = [batch_size, dec_hid_dim]
        return enc_output, enc_hidden, s


class Decoder(nn.Module):
    def __init__(self, attention, output_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.GRU(enc_hid_dim * 2, dec_hid_dim, bidirectional=True)
        self.fc_out = nn.Linear((enc_hid_dim + dec_hid_dim) * 2, output_dim)
        self.fc = nn.Linear(dec_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, enc_output, enc_hidden):
        a = self.attention(s, enc_output).unsqueeze(1)
        # a = [batch_size, 1, src_len]

        enc_output = enc_output.transpose(0, 1)
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]

        c = torch.bmm(a, enc_output).transpose(0, 1)
        # c = [1, batch_size, enc_hid_dim * 2]

        dec_output, dec_hidden = self.rnn(c, enc_hidden)
        # dec_output = [src_len(=1), batch_size, dec_hid_dim * num_directions(=2)] = [1, 64, 256]
        # dec_hidden = [n_layers * num_directions(=2), batch_size, dec_hid_dim] = [2, 64, 128]

        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        # dec_output = [batch_size, dec_hid_dim * 2]
        # c = [batch_size, enc_hid_dim * 2]

        pred = self.fc_out(torch.cat((dec_output, c), dim=1))
        # pred = [batch_size, output_dim]

        s = torch.tanh(self.fc(torch.cat((dec_hidden[-2, :, :], dec_hidden[-1, :, :]), dim=1)))
        return pred, dec_hidden, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        src_len = enc_output.shape[0]

        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        # energy = [batch_size, src_len, dec_hid_dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch_size, src_len]

        return F.softmax(attention, dim=1)


class Seq2Seq_with_attention(nn.Module):
    def __init__(self, input_dim, output_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Seq2Seq_with_attention, self).__init__()
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.encoder = Encoder(input_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.decoder = Decoder(self.attention, output_dim, enc_hid_dim, dec_hid_dim, dropout)

    def forward(self, src):
        enc_output, enc_hidden, s = self.encoder(src)
        pred, dec_hidden, s = self.decoder(s, enc_output, enc_hidden)
        return pred


# dataset
class TrainSet(Dataset):
    def __init__(self, datax, datay):
        self.data, self.label = datax, datay

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

