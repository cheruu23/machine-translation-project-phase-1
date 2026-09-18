"""
nmt_translator.py
-----------------
Self-contained inference module for the English -> Amharic NMT project.

Contains:
  * the Basic Seq2Seq + LSTM architecture
  * the Attention Seq2Seq + LSTM architecture
  * greedy decoding for both
  * a Translator class that loads tokenizers + checkpoints once and
    exposes .translate(text, model="attention")

The architecture definitions here MUST match the ones used at training time
(EMB_DIM=96, HID_DIM=192, N_LAYERS=1, DROPOUT=0.20, MAX_LEN=32, vocab=7000).
"""

from pathlib import Path
import re
import unicodedata

import torch
import torch.nn as nn
import sentencepiece as spm

# --------------------------------------------------------------------------
# Special token ids (fixed at SentencePiece training time)
# --------------------------------------------------------------------------
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# Architecture constants used by the saved checkpoints
EMB_DIM = 96
HID_DIM = 192
N_LAYERS = 1
DROPOUT = 0.20
MAX_LEN = 32


# --------------------------------------------------------------------------
# Preprocessing (identical to the training notebook)
# --------------------------------------------------------------------------
def normalize_english(text: str) -> str:
    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_amharic(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text).strip())
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*\u1362\s*", "\u1362 ", text).strip()


# --------------------------------------------------------------------------
# Basic Seq2Seq + LSTM
# --------------------------------------------------------------------------
class EncoderBasic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class DecoderBasic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell):
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc_out(output.squeeze(1)), hidden, cell


class Seq2SeqBasic(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        import random
        hidden, cell = self.encoder(src)
        token = tgt[:, 0]
        outputs = []
        for t in range(1, tgt.shape[1]):
            output, hidden, cell = self.decoder(token, hidden, cell)
            outputs.append(output)
            top1 = output.argmax(1)
            token = tgt[:, t] if random.random() < teacher_forcing_ratio else top1
        return torch.stack(outputs, dim=1)


# --------------------------------------------------------------------------
# Attention Seq2Seq + LSTM
# --------------------------------------------------------------------------
class EncoderAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=True,
        )
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        n = hidden.shape[0] // 2
        hidden = torch.cat([hidden[:n], hidden[n:]], dim=2)
        cell = torch.cat([cell[:n], cell[n:]], dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[1]
        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)


class DecoderAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            hid_dim * 2 + emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        attention = self.attention(hidden[-1], encoder_outputs, mask)
        weighted = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(
            torch.cat(
                (output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)),
                dim=1,
            )
        )
        return prediction, hidden, cell, attention


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        import random
        encoder_outputs, hidden, cell = self.encoder(src)
        mask = src != PAD_ID
        token = tgt[:, 0]
        outputs = []
        for t in range(1, tgt.shape[1]):
            output, hidden, cell, _ = self.decoder(
                token, hidden, cell, encoder_outputs, mask
            )
            outputs.append(output)
            top1 = output.argmax(1)
            token = tgt[:, t] if random.random() < teacher_forcing_ratio else top1
        return torch.stack(outputs, dim=1)


# --------------------------------------------------------------------------
# Translator façade used by the API / UI
# --------------------------------------------------------------------------
class Translator:
    """Loads tokenizers + checkpoints once; exposes translate()."""

    def __init__(self, artifact_dir="artifacts", device=None,
                 basic_ckpt="basic_seq2seq_best.pt",
                 attn_ckpt="attention_seq2seq_best.pt"):
        self.dir = Path(artifact_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.sp_en = spm.SentencePieceProcessor(
            model_file=str(self.dir / "en_tokenizer.model")
        )
        self.sp_am = spm.SentencePieceProcessor(
            model_file=str(self.dir / "am_tokenizer.model")
        )
        self.src_vocab = self.sp_en.get_piece_size()
        self.tgt_vocab = self.sp_am.get_piece_size()

        # Attention model (primary deployed model)
        self.attn = Seq2SeqAttn(
            EncoderAttn(self.src_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
            DecoderAttn(self.tgt_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
            self.device,
        ).to(self.device)
        ckpt = torch.load(self.dir / attn_ckpt, map_location=self.device,
                          weights_only=False)
        self.attn.load_state_dict(ckpt["model_state"])
        self.attn.eval()
        self.attn_epoch = ckpt.get("epoch")

        # Basic model (optional, for side-by-side demo)
        self.basic = None
        basic_path = self.dir / basic_ckpt
        if basic_path.exists():
            self.basic = Seq2SeqBasic(
                EncoderBasic(self.src_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
                DecoderBasic(self.tgt_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
                self.device,
            ).to(self.device)
            bck = torch.load(basic_path, map_location=self.device,
                             weights_only=False)
            self.basic.load_state_dict(bck["model_state"])
            self.basic.eval()

    # -- decoding -----------------------------------------------------------
    def _encode_src(self, sentence):
        ids = ([BOS_ID]
               + self.sp_en.encode(normalize_english(sentence), out_type=int)[:MAX_LEN]
               + [EOS_ID])
        return ids, torch.tensor(ids, dtype=torch.long,
                                 device=self.device).unsqueeze(0)

    @torch.inference_mode()
    def _greedy_basic(self, sentence, max_len=MAX_LEN):
        _, src = self._encode_src(sentence)
        hidden, cell = self.basic.encoder(src)
        token = torch.tensor([BOS_ID], dtype=torch.long, device=self.device)
        out = []
        for _ in range(max_len):
            logits, hidden, cell = self.basic.decoder(token, hidden, cell)
            nxt = logits.argmax(1).item()
            if nxt == EOS_ID:
                break
            if nxt not in (PAD_ID, BOS_ID):
                out.append(nxt)
            token = torch.tensor([nxt], dtype=torch.long, device=self.device)
        return self.sp_am.decode(out)

    @torch.inference_mode()
    def _greedy_attn(self, sentence, max_len=MAX_LEN, return_attention=False,
                     no_repeat_window=0):
        ids, src = self._encode_src(sentence)
        enc_out, hidden, cell = self.attn.encoder(src)
        mask = src != PAD_ID
        token = torch.tensor([BOS_ID], dtype=torch.long, device=self.device)
        out, rows = [], []
        for _ in range(max_len):
            logits, hidden, cell, att = self.attn.decoder(
                token, hidden, cell, enc_out, mask
            )
            if return_attention:
                rows.append(att[0].float().cpu().numpy())
            # optional anti-loop guard for the demo UI
            if no_repeat_window and len(out) >= no_repeat_window:
                for t in set(out[-no_repeat_window:]):
                    logits[0, t] = torch.finfo(logits.dtype).min
            nxt = logits.argmax(1).item()
            if nxt == EOS_ID:
                break
            if nxt not in (PAD_ID, BOS_ID):
                out.append(nxt)
            token = torch.tensor([nxt], dtype=torch.long, device=self.device)
        text = self.sp_am.decode(out)
        if return_attention:
            import numpy as np
            return text, np.array(rows), ids
        return text

    # -- public API ---------------------------------------------------------
    def translate(self, text, model="attention", no_repeat_window=0):
        if not text or not str(text).strip():
            return ""
        if model == "basic":
            if self.basic is None:
                raise ValueError("Basic checkpoint not available.")
            return self._greedy_basic(text)
        return self._greedy_attn(text, no_repeat_window=no_repeat_window)

    def translate_both(self, text):
        return {
            "attention": self.translate(text, "attention"),
            "basic": self.translate(text, "basic") if self.basic else None,
        }

    def attention_matrix(self, text):
        """Returns (translation, weights[T_tgt, T_src], source_pieces)."""
        text_out, weights, ids = self._greedy_attn(text, return_attention=True)
        pieces = [self.sp_en.id_to_piece(i) for i in ids]
        return text_out, weights, pieces
