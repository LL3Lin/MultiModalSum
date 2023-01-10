import logging
import os

import torch

from data import Vocab, DataLoader, BOS, EOS
from generator import MemGenerator
from optim import Adam, get_inverse_sqrt_schedule_with_warmup
from utils import move_to_device, set_seed, Statistics
from work import validate

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # build dict
    vocabs = Vocab(args["vocab"], 0, [BOS, EOS])
    logger.info(args)
    logger.info("vocab, size %d, coverage %.3f", vocabs.size, vocabs.coverage)

    # set seed
    set_seed(2022)

    # set model
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)

    model = MemGenerator(vocabs, args["embed_dim"], args["ff_embed_dim"], args["num_heads"], args["dropout"],
                         args["enc_layers"], args["dec_layers"], args["label_smoothing"])

    global_step = 0
    model = model.to(device)
    optimizer = Adam([{'params': [p for p in model.parameters() if p.requires_grad],
                       'lr': args['embed_dim'] ** -0.5}],
                     betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_inverse_sqrt_schedule_with_warmup(optimizer, args['warmup_steps'], args['total_train_steps'])

    # prepare train data
    train_data = DataLoader(vocabs, args['train_data'], args['train_batch_size'], for_train=True)

    # begin training
    step, epoch = 0, 0
    tr_stat = Statistics()
    logger.info("start training")
    model.train()
    while global_step <= args['total_train_steps']:
        for batch in train_data:
            batch = move_to_device(batch, device)
            loss, acc = model(batch)

            tr_stat.update({'loss': loss.item() * batch['tgt_num_tokens'],
                            'tokens': batch['tgt_num_tokens'],
                            'acc': acc})

            tr_stat.step()
            loss.backward()
            step += 1
            if not (step % args['gradient_accumulation_steps'] == 0):
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args['print_every'] == 0:
                logger.info("epoch %d, step %d, loss %.3f, acc %.3f", epoch, global_step,
                            tr_stat['loss'] / tr_stat['tokens'], tr_stat['acc'] / tr_stat['tokens'])
                tr_stat = Statistics()

            # begin validating
            if global_step % args['eval_every'] == 0:
                model.eval()
                max_time_step = 256 if global_step > 2 * args['warmup_steps'] else 5
                devbleus = []
                dev_data = DataLoader(vocabs, args['dev_data'], args['dev_batch_size'], for_train=False)
                devbleu = validate(device, model, dev_data, beam_size=1, alpha=0.6, max_time_step=max_time_step)
                devbleus.append(devbleu)
                devbleu = sum(devbleus) / len(devbleus)
                logger.info("epoch %d, step %d, dev bleu %.2f", epoch, global_step, devbleu)
                torch.save({'args': args, 'model': model.state_dict()}, '%s/best.pt' % (args['ckpt']))
                if not args['only_save_best']:
                    torch.save({'args': args, 'model': model.state_dict()},
                               '%s/epoch%d_batch%d_devbleu%.2f' % (args['ckpt'], epoch, global_step, devbleu))

                model.train()
            if global_step > args['total_train_steps']:
                break
        epoch += 1
    logger.info('finish training after %d steps', global_step)


if __name__ == "__main__":

    args = {
        "vocab": '../vocab.bpe',
        "embed_dim": 512,
        "ff_embed_dim": 2048,
        "num_heads": 8,
        "enc_layers": 6,
        "dec_layers": 6,
        "dropout": 0.1,
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 2,
        "total_train_steps": 1000000,
        "warmup_steps": 4500,
        "train_batch_size": 1024,
        "dev_batch_size": 1024,
        "resume_ckpt": False,
        "train_data": '../train.bpe',
        "dev_data": '../dev.bpe',
        "ckpt": 'ckpt',
        "print_every": 200,
        "eval_every": 5000,
        "only_save_best": False
    }

    if not os.path.exists(args["ckpt"]):
        os.mkdir(args["ckpt"])

    main(args)
