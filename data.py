# -*- coding: utf-8 -*-
"""Data code."""

import torch
import logging
import spacy
from datasets import load_dataset
from spacy.symbols import ORTH
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator


# special tokens
PAD_TOKEN = "<pad>"
INIT_TOKEN = "<sos>"
END_TOKEN = "<eos>"
SP_TOKENS = ["<unk>", PAD_TOKEN, INIT_TOKEN, END_TOKEN]


class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset."""

    def __init__(self, items, data='multi30k', task='en2de', pad_index=1):
        """Init."""
        self.items = items
        self.data = data
        self.task = task
        self.pad_index = pad_index

    def __len__(self):
        """Length."""
        return len(self.items)

    def __getitem__(self, index):
        """Get item."""
        item = self.items[index]

        if self.data == 'multi30k':
            en = item['en']
            de = item['de']
            if self.task == 'en2de':
                src, dst = en, de

            elif self.task == 'de2en':
                src, dst = de, en

            return src, dst

    def set_tokenizers(self, src_tokenizer, dst_tokenizer):
        """Set tokenizers."""
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer

    def set_vocabs(self, src_vocab, dst_vocab):
        """Set vocabs."""
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab

    def collate_fn(self, batch):
        """Collate function for dataloader."""
        # split two sentences
        src, dst = list(zip(*batch))

        # add special tokens
        src = ['<sos> ' + sentecne.lower() + ' <eos>' for sentecne in src]
        dst = ['<sos> ' + sentecne.lower() + ' <eos>' for sentecne in dst]

        # make torch tensor from sentences
        # 1. sentence to tokens
        # 2. token to index of vocab
        # 3. index of list to torch.tensor
        src_batch = sentence2index(src,
                                   self.src_tokenizer,
                                   self.src_vocab,
                                   self.pad_index)
        dst_batch = sentence2index(dst,
                                   self.dst_tokenizer,
                                   self.dst_vocab,
                                   self.pad_index)

        return src_batch, dst_batch


def sentence2index(sentences, tokenizer, vocab, pad_index=1):
    """Make a torch.tensor with list of sentences."""
    indices = []
    max_len = 0
    for sentence in sentences:
        temp = []
        for token in tokenizer(sentence):
            temp.append(vocab[token.text])
        indices.append(temp)
        max_len = max(max_len, len(temp))
    batch = []
    for item in indices:
        item += [pad_index] * (max_len - len(item))
        batch.append(item)
    return torch.tensor(batch)


def get_tokenizers(data='multi30k', task='en2de'):
    """Get tokenizers."""
    if data == 'multi30k':
        spacy_en = spacy.load('en_core_web_sm')
        spacy_de = spacy.load('de_core_news_sm')

        for token in SP_TOKENS:
            spacy_en.tokenizer.add_special_case(token, [{ORTH: token}])
            spacy_de.tokenizer.add_special_case(token, [{ORTH: token}])

        en_tokenizer = spacy_en.tokenizer
        de_tokenizer = spacy_de.tokenizer

        if task == 'en2de':
            res = {'src': en_tokenizer, 'dst': de_tokenizer}
        elif task == 'de2en':
            res = {'src': de_tokenizer, 'dst': en_tokenizer}
        else:
            raise NotImplementedError("Not expected task")

    return res


def build_vocab(sentence_iter, tokenizers, data='multi30k', min_freq=2):
    """Build vocaburary with sentecne iterator."""
    def yield_tokens(sentence_iter, tokenizer, key):
        for sentences in sentence_iter:
            sentence = sentences[key]
            tokens = tokenizer(sentence)
            yield [token.text.lower() for token in tokens]

    src_tokenizer = tokenizers['src']
    dst_tokenizer = tokenizers['dst']

    src_vocab = build_vocab_from_iterator(
        yield_tokens(sentence_iter, src_tokenizer, 0),
        specials=SP_TOKENS, min_freq=min_freq)
    src_vocab.set_default_index(src_vocab["<unk>"])

    dst_vocab = build_vocab_from_iterator(
        yield_tokens(sentence_iter, dst_tokenizer, 1),
        specials=SP_TOKENS, min_freq=min_freq)
    dst_vocab.set_default_index(src_vocab["<unk>"])

    res = {'src': src_vocab, 'dst': dst_vocab}

    return res


def get_text_data(data='multi30k', task='en2de', pad_index=1, min_freq=2):
    """Get dataset, tokenizer and vocab."""
    # load files
    if data == 'multi30k':
        ds = load_dataset('bentrevett/multi30k')

    train_dataset = CustomDataset(ds['train'], data, task, pad_index)
    valid_dataset = CustomDataset(ds['validation'], data, task, pad_index)
    test_dataset = CustomDataset(ds['test'], data, task, pad_index)

    # load tokenizers
    tokenizers = get_tokenizers(data, task)

    # build vocabularies from train dataset
    vocabs = build_vocab(train_dataset, tokenizers, data, min_freq)
    logging.info(("Build vocabularies from train dataset."
                  f"\n\t\t\t - Number of src vocab: {len(vocabs['src'])}"
                  f"\n\t\t\t - Number of dst vocab: {len(vocabs['dst'])}"
                  f"\n\t\t\t - Src Tokens(0~9): {vocabs['src'].lookup_tokens(range(10))}"
                  f"\n\t\t\t - Dst Tokens(0~9): {vocabs['dst'].lookup_tokens(range(10))}"))

    # set tokenizer and vocab to make batch input
    train_dataset.set_tokenizers(tokenizers['src'], tokenizers['dst'])
    valid_dataset.set_tokenizers(tokenizers['src'], tokenizers['dst'])
    train_dataset.set_vocabs(vocabs['src'], vocabs['dst'])
    valid_dataset.set_vocabs(vocabs['src'], vocabs['dst'])

    datasets = {'train': train_dataset,
                'valid': valid_dataset,
                'test': test_dataset}

    return tokenizers, vocabs, datasets


if __name__ == '__main__':
    from pathlib import Path
    data_root = Path('./dataset/multi30k')
    _, vocabs, train_dataset, valid_dataset, test_dataset = get_text_data(data_root)
    src_vocab, dst_vocab = vocabs['src'], vocabs['dst']
    print(f"Number of vocab (src): {len(src_vocab)}")
    print(f"Number of vocab (dst): {len(dst_vocab)}")

    print(("Number of train/valid/test dataset: "
           f"{len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}"))
