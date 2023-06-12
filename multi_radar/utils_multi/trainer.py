"""Training process
"""  
import torch
import torch.nn as nn
import tqdm
from utils_multi.result_utils import *

class Trainer:

    def __init__(self, model, data_train, data_valid, data_test, args, device):
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.args = args
        self.device = device

        if self.args.train.early_stop:
            self.min_val_loss = None
            self.steps_waited = 0

    def train(self):
        self.model = self.model.to(self.device)
        loss_fn=nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.train.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epoch)

        Epoch_num = self.args.train.epoch
        step = 0
        for epoch in range(Epoch_num):
            self.model.train()
            progress_bar = tqdm.tqdm(self.data_train)
            for iter, (x_batch, y_batch) in enumerate(progress_bar):
                x_batch = x_batch.to(self.device, dtype=torch.float)
                y_batch = y_batch.to(self.device, dtype=torch.float)    
            
                y_batch_pred = self.model(x_batch)
                loss = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print
                step += 1
                progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Total loss: {:.5f}.'.format(
                    step, epoch+1, Epoch_num, iter + 1, len(self.data_train), loss.item()))

            valid_acc, valid_loss = test(self.data_valid, self.device, self.model, loss_fn)
            print('valid acc ', valid_acc, 'valid_loss ', valid_loss)

            test_acc, test_loss = test(self.data_test, self.device, self.model, loss_fn)
            print('test acc ', test_acc, 'test_loss ', test_loss)
            
            if self.args.train.early_stop:
                if self.min_val_loss is None:
                    self.min_val_loss = valid_loss

                elif valid_loss >= self.min_val_loss - self.args.train.delta:
                    self.steps_waited += 1
                    print('epoch', epoch+1, 'steps waited', self.steps_waited)

                    if self.steps_waited >= self.args.train.patience:
                        print('break the training loop and saving results-------')
                        break
                
                else:
                    self.min_val_loss = valid_loss
                    self.steps_waited = 0

                    # save the best model
                    print(epoch+1, 'update min loss')
                    cm_set = (self.data_valid, self.data_test)
                    cm_set_names = ('valid', 'test')
                    for set, set_name in zip(cm_set, cm_set_names):
                        save_result(
                        set=set,
                        set_name='best-'+set_name,
                        device=self.device,
                        model=self.model,
                        args=self.args,
                        )


            lr_scheduler.step()