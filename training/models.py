import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_p=0.0):
        super().__init__()
        # LSTM internal dropout only applies if num_layers > 1
        lstm_dropout = dropout_p if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers,
                            bidirectional=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(p=dropout_p)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        hidden = self.dropout(hidden)
        return self.proj(hidden)
