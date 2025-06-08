import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch.nn.functional as F

def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        # Layers to training
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # For encoder hidden states
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # For decoder hidden states
        self.v = nn.Linear(hidden_size, 1, bias=False)  # To compute scalar attention scores

        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """
        batch_size, max_tgt_len, hidden_size = query.size()
        max_src_len = encoder_outputs.size(1)

        query_expanded = query.unsqueeze(2).expand(-1, -1, max_src_len, -1)
        encoder_outputs_expanded = encoder_outputs.unsqueeze(1).expand(-1, max_tgt_len, -1, -1)

        # Attention scores
        scores = self.v(
            torch.tanh(
                self.W_h(encoder_outputs_expanded) + self.W_s(query_expanded)
            )
        ).squeeze(-1)  # Shape: (batch_size, max_tgt_len, max_src_len)

        mask = ~self.sequence_mask(src_lengths)  # True for padding positions
        mask = mask.unsqueeze(1).expand(-1, max_tgt_len, -1)  # Shape: (batch_size, max_tgt_len, max_src_len)

        scores = scores.masked_fill(mask, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, max_tgt_len, max_src_len)

        attn_out = torch.bmm(attn_weights, encoder_outputs)  # Shape: (batch_size, max_tgt_len, hidden_size)


        combined = torch.cat((attn_out, query), dim=-1)  # Shape: (batch_size, max_tgt_len, 2 * hidden_size)
        attn_out = torch.tanh(self.linear_out(combined))  # Shape: (batch_size, max_tgt_len, hidden_size)

        return attn_out, attn_weights

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):

        embedded = self.embedding(src)  # (batch_size, max_src_len, hidden_size)
        embedded = self.dropout(embedded)

        packed_embedded = pack(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )


        packed_output, (h_n, c_n) = self.lstm(packed_embedded)

        # Unpack the sequence
        enc_output, _ = unpack(
            packed_output,
            batch_first=True
        )  # (batch_size, max_src_len, hidden_size)
        
        enc_output = self.dropout(enc_output)

        return enc_output, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        embedded = self.embedding(tgt)  # (batch_size, max_tgt_len, hidden_size)
        embedded = self.dropout(embedded)

        output, dec_state = self.lstm(embedded, dec_state)  # output: (batch_size, max_tgt_len, hidden_size)

        # Apply attention
        if self.attn is not None:
            output, attn_weights = self.attn(output, encoder_outputs, src_lengths)
        
        
        output = self.dropout(output)


        return output, dec_state


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)
        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):
        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
