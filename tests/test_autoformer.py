from dataclasses import dataclass, field
import sys
import time
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader

from forecast.data.dataset import SNDLibDataset
from forecast.model.autoformer import Autoformer
from forecast.utils.cmdparser import HfArgumentParser, ModelArguments, DataArguments

device = "cuda:5"

parser = HfArgumentParser((ModelArguments, DataArguments))
args, data_args = parser.parse_args_into_dataclasses()

train_data = SNDLibDataset(
    path=data_args.data_path,
    size=(args.seq_len,args.label_len, args.pred_len)
)
train_data.save_data()

model = Autoformer(args).float().to(device)

optim = Adam(model.parameters(), lr=args.learning_rate)
loss_func = nn.MSELoss()

train_loader = DataLoader(train_data, batch_size=32)

train_steps = len(train_loader)


for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        # print(batch_x.size(), batch_x_mark.size(), batch_y.shape, batch_y_mark.size())
        optim.zero_grad()

        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # pred, true = _process_one_batch(
        #     train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
        # )

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        loss = loss_func(outputs, batch_y)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                    i + 1, epoch + 1, loss.item()
                )
            )
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        loss.backward()
        optim.step()
