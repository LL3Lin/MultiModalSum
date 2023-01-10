import logging
import random

import numpy as np


PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        num_tot_tokens = 0
        num_invocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.rstrip('\n').split(' ')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                logger.info("(Vocab)Illegal line:", line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_invocab_tokens += cnt
        self.coverage = num_invocab_tokens / num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)


def ListsToTensor(xs, vocab=None, worddrop=0., local_vocabs=None):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < worddrop:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad] * (max_len - len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data


def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
    return data


def batchify(data, vocabs, max_src_len, max_tgt_len):
    src_tokens = [x['src_tokens'][:max_src_len] for x in data]
    ast_tokens = [x['ast_tokens'][:max_src_len] for x in data]
    mem_tokens = [x['mem_tokens'][:max_tgt_len] for x in data]
    tgt_tokens_in = [[BOS] + x['tgt_tokens'][:max_tgt_len] for x in data]
    tgt_tokens_out = [x['tgt_tokens'][:max_tgt_len] + [EOS] for x in data]

    src_token = ListsToTensor(src_tokens, vocabs)
    ast_token = ListsToTensor(ast_tokens, vocabs)
    mem_token = ListsToTensor(mem_tokens, vocabs)

    tgt_token_in = ListsToTensor(tgt_tokens_in, vocabs)
    tgt_token_out = ListsToTensor(tgt_tokens_out, vocabs)

    not_padding = (tgt_token_out != vocabs.padding_idx).astype(np.int64)
    tgt_lengths = np.sum(not_padding, axis=0)
    tgt_num_tokens = int(np.sum(tgt_lengths))

    ret = {
        'src_tokens': src_token,
        'ast_tokens': ast_token,
        'mem_tokens': mem_token,
        'tgt_tokens_in': tgt_token_in,
        'tgt_tokens_out': tgt_token_out,
        'tgt_num_tokens': tgt_num_tokens,
        'tgt_raw_sents': [x['tgt_tokens'] for x in data],
        'indices': [x['index'] for x in data]
    }
    return ret


class DataLoader(object):
    def __init__(self, vocabs, filename, batch_size, for_train, max_src_len=110, max_tgt_len=18):
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train

        src_tokens, ast_tokens, tgt_tokens, mem_tokens = [], [], [], []
        src_sizes, tgt_sizes = [], []
        for line in open(filename, 'r', encoding='utf-8').readlines():
            try:
                src, ast, tgt, mem = line.strip().split('\t')
            except:
                continue
            src, ast, tgt, mem = src.split(), ast.split(), tgt.split(), mem.split()

            src_sizes.append(len(src))
            tgt_sizes.append(len(tgt))
            src_tokens.append(src)
            ast_tokens.append(ast)
            tgt_tokens.append(tgt)
            mem_tokens.append(mem)

        self.src = src_tokens
        self.ast = ast_tokens
        self.tgt = tgt_tokens
        self.mem = mem_tokens
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        logger.info("read %s file with %d paris. max src len: %d, max tgt len: %d", filename,
                    len(self.src), self.src_sizes.max(), self.tgt_sizes.max())

    def __len__(self):
        return len(self.src)

    def __iter__(self):
        if self.train:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

        batches = []
        num_tokens, batch = 0, []
        for i in indices:
            num_tokens += 1 + max(self.src_sizes[i], self.tgt_sizes[i])
            if num_tokens > self.batch_size:
                batches.append(batch)
                num_tokens, batch = 1 + max(self.src_sizes[i], self.tgt_sizes[i]), [i]
            else:
                batch.append(i)

        if not self.train or num_tokens > self.batch_size / 2:
            batches.append(batch)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            data = []
            for i in batch:
                src_tokens = self.src[i]
                ast_tokens = self.ast[i]
                tgt_tokens = self.tgt[i]
                mem_tokens = self.mem[i]
                item = {'src_tokens': src_tokens, 'ast_tokens': ast_tokens,
                        'tgt_tokens': tgt_tokens, 'mem_tokens': mem_tokens,
                        'index': i}
                data.append(item)
            yield batchify(data, self.vocabs, self.max_src_len, self.max_tgt_len)
