from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD
from decoder import decode
from utils import concat_inputs

from dataloader import get_dataloader

def train(model, args):
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    dev_loader = get_dataloader(args.dev_json, args.batch_size, False)
    criterion = CTCLoss(zero_infinity=True)
    #optimiser = SGD(model.parameters(), lr=args.lr)
    USE_ADAM = False
    if USE_ADAM:
       optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
       optimiser = SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=False
)

    def train_one_epoch(epoch):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('checkpoints/{}'.format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_per = 200

    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_dev_loss = 0.
        for idx, data in enumerate(dev_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            outputs = log_softmax(model(inputs), dim=-1)
            dev_loss = criterion(outputs, targets, in_lens, out_lens)
            running_dev_loss += dev_loss
        avg_dev_loss = running_dev_loss / len(dev_loader)
        scheduler.step(avg_dev_loss)
        print(f"current LR: {optimiser.param_groups[0]['lr']}")
        dev_decode = decode(model, args, args.dev_json)
        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, avg_dev_loss, dev_decode[4])
            )

        if dev_decode[4] < best_per:
            best_per = dev_decode[4]
            model_path = 'checkpoints/{}/model_{}'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
            
    return model_path
