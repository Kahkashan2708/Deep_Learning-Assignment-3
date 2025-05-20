import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim=128, hidden_size=256, 
                 rnn_type='LSTM', num_layers=1, device='cpu'):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(input_vocab_size, emb_dim)
        self.target_embedding = nn.Embedding(output_vocab_size, emb_dim)

        rnn_cls = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }[self.rnn_type]

        self.encoder = rnn_cls(input_size=emb_dim, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True)

        self.decoder = rnn_cls(input_size=emb_dim, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, source, target):
        
        embedded_src = self.embedding(source)
        encoder_outputs, hidden = self.encoder(embedded_src)

        embedded_tgt = self.target_embedding(target)
        decoder_outputs, _ = self.decoder(embedded_tgt, hidden)
        output = self.output_layer(decoder_outputs)

        return output  