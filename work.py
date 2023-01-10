import json
import logging
import re
import time

import numpy as np
import sacrebleu
import torch

from data import Vocab, DataLoader, BOS, EOS
from generator import MemGenerator
from utils import move_to_device

logger = logging.getLogger(__name__)


def generate_batch(model, batch, beam_size, alpha, max_time_step):
    token_batch = []
    beams = model.work(batch, beam_size, max_time_step)
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_token = [token for token in best_hyp.seq[1:-1]]
        token_batch.append(predicted_token)
    return token_batch, batch['indices']


def validate(device, model, test_data, beam_size=5, alpha=0.6, max_time_step=100, dump_path=None):
    """For Development Only"""

    ref_stream = []
    sys_stream = []
    for batch in test_data:
        batch = move_to_device(batch, device)
        res, _ = generate_batch(model, batch, beam_size, alpha, max_time_step)
        sys_stream.extend(res)
        ref_stream.extend(batch['tgt_raw_sents'])

    assert len(sys_stream) == len(ref_stream)

    sys_stream = [re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in sys_stream]
    ref_stream = [re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in ref_stream]
    ref_streams = [ref_stream]

    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams,
                                 force=True, lowercase=False,
                                 tokenize='none').score

    if dump_path is not None:
        results = {'sys_stream': sys_stream,
                   'ref_stream': ref_stream}
        json.dump(results, open(dump_path, 'w'))
    return bleu


if __name__ == "__main__":

    args = {
        'load_path': 'ckpt/',
        'vocab_path': '../vocab.bpe',
        'test_data': '../test.bpe',
        'output_path': '../',
        'test_batch_size': 8192,
        'beam_size': 1,
        'alpha': 0.6,
        'max_time_step': 256,
        'device': 0,
        'retain_bpe': False,
        'comp_bleu': False,
        # Only for debug and analyze
        'dump_path': None
    }

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    test_model = args['load_path']
    model_args = torch.load(args['load_path'])['args']
    print(model_args)

    vocabs = Vocab(args['vocab_path'], 0, [BOS, EOS])

    if args['device'] < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args['device'])

    model = MemGenerator(vocabs, model_args['embed_dim'], model_args['ff_embed_dim'], model_args['num_heads'],
                         model_args['dropout'], model_args['enc_layers'], model_args['dec_layers'],
                         model_args['label_smoothing'])

    test_data = DataLoader(vocabs, args['test_data'], args['test_batch_size'], for_train=False)

    model.load_state_dict(torch.load(test_model)['model'])
    model = model.to(device)
    model.eval()
    if args['comp_bleu']:
        bleu = validate(device, model, test_data, beam_size=args['beam_size'], alpha=args['alpha'],
                        max_time_step=args['max_time_step'], dump_path=args['dump_path'])
        logger.info("%s %s %.2f", test_model, args['test_data'], bleu)

    if args['output_path'] is not None:
        start_time = time.time()
        TOT = len(test_data)
        DONE = 0
        logger.info("%d/%d", DONE, TOT)
        outs, indices = [], []
        for batch in test_data:
            batch = move_to_device(batch, device)
            res, ind = generate_batch(model, batch, args['beam_size'], args['alpha'], args['max_time_step'])
            for out_tokens, index in zip(res, ind):
                if args['retain_bpe']:
                    out_line = ' '.join(out_tokens)
                else:
                    out_line = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(out_tokens))
                DONE += 1
                if DONE % 10000 == -1 % 10000:
                    logger.info("%d/%d", DONE, TOT)
                outs.append(out_line)
                indices.append(index)
        end_time = time.time()
        logger.info("Time elapsed: %f", end_time - start_time)
        order = np.argsort(np.array(indices))
        with open(args['output_path'], 'w') as fo:
            for i in order:
                out_line = outs[i]
                fo.write(out_line + '\n')
