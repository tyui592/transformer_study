# -*- coding: utf-8 -*-
"""Evaluation Code."""

import torch
from data import sentence2index, INIT_TOKEN, END_TOKEN
from torchtext.data.metrics import bleu_score


def translate(src: str,
              model,
              src_tokenizer,
              src_vocab,
              dst_vocab,
              pad_index,
              max_len,
              device):
    """Translate a source sentence."""
    x = sentence2index([src], src_tokenizer, src_vocab, pad_index).to(device)

    model.eval()
    with torch.no_grad():
        memory = model.encoder(x)

        # autoregressive sentence generation with sos token
        indices = [dst_vocab[INIT_TOKEN]]
        for _ in range(max_len):
            dec_x = torch.LongTensor(indices).unsqueeze(0).to(device)
            dec_mask = model.make_dec_mask(dec_x)
            output, attention = model.decoder(dec_x, memory, dec_mask)
            pred_id = output.argmax(2)[:, -1].item()
            indices.append(pred_id)
            if pred_id == dst_vocab[END_TOKEN]:
                break
    translated_tokens = dst_vocab.lookup_tokens(indices)
    return translated_tokens[1:-1], attention


def calc_bleu_score(sentence_iter, model, src_tokenizer, dst_tokenizer,
                    src_vocab, dst_vocab, device, pad_index, max_len=50):
    """Calculate a BLEU score."""
    # list for each corpus
    candidates, references = [], []

    for src, dst in sentence_iter:
        pred, _ = translate(src, model, src_tokenizer,
                            src_vocab, dst_vocab, pad_index, max_len, device)

        candidates.append(pred)
        references.append([[token.text.lower() for token in dst_tokenizer(dst)]])

    bleu = bleu_score(candidates, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

    return bleu, (candidates[0], references[0][0])


def evaluate_step(model, dataloader, criterion, d_out, device):
    """Evaluate a model."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, dst in dataloader:
            src, dst = src.to(device), dst.to(device)

            output, _ = model(src, dst[:, :-1])

            output = output.contiguous().view(-1, d_out)
            dst = dst[:, 1:].contiguous().view(-1)

            loss = criterion(output, dst)

            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
