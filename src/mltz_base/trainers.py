import torch
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NNTrainer:
    
    SUPPORTED_DEVICES = {'cpu','cuda','cuda:2','mps'}
    
    def __init__(self,device = 'cpu',
                 model = None,
                 loss_function = None,
                 optimizer = None,
                 scoring_functions = {},
                 console_verbose = True,
                 tensor_board_verbose = True,
                 log_dir='_tb_log_dir'):
        self.device = device
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scoring_functions = scoring_functions
        self.scores = defaultdict(list)
        self.console_verbose = console_verbose
        self.tensor_board_verbose = tensor_board_verbose
        self.writer = SummaryWriter(log_dir)

    def with_device(self, device):
        if device in NNTrainer.SUPPORTED_DEVICES:
            self.device = device
        else: 
            raise TypeError(f'{device} is not supported option config in set_device()')
        return self

    def with_loss_function(self, loss_function):
        self.loss_function = loss_function
        return self
    
    def with_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self
    
    def with_model(self, model):
        self.model = model
        return self
    
    def with_scoring_functions(self,scoring_functions):
        self.scoring_functions = scoring_functions
        return self
    
    def with_console_verbose(self, console_verbose):
        self.console_verbose = console_verbose
        return self
    
    def report_scores (self, 
                      y_true,            # ground truth 
                      y_pred,            # prediction
                      prefix,            # label/namespace
                      loss,              # loss
                      epoch_num,         # which epoch we're on
                      current_in_batch,  # how many observations processed
                      total_in_batch,    # how many observations in whole batch
                      step):             # which step of training/evaluation we're on
        """
        This function reports the scores of a neural network training.
        """
        def report_to_tensorboard(name, value, step):
            if self.tensor_board_verbose:
                namespace = f'{self.model.short_name}/{prefix}/{name}'
                self.writer.add_scalar(namespace, value, step)
                self.writer.flush()
        
        def report_loss():
            self.scores[f'{prefix}_loss'].append(loss)
            report_to_tensorboard('loss', loss, step)
            return f"loss: {loss:>7f}"
        
        def report_scores():
            scores_text = ""
            for score_name, score_func in self.scoring_functions.items():
                score_observation = score_func(y_true, y_pred)
                self.scores[f'{prefix}_{score_name}'].append(score_observation)
                report_to_tensorboard(score_name, score_observation, step)
                scores_text += f" {score_name} {str(score_observation)}"
            return scores_text

        header = f"In {prefix} [{current_in_batch:>5d}/{total_in_batch:>5d}] "
        body = f"epoch: {epoch_num} " + report_loss() + report_scores()

        if self.console_verbose:
            print(header + body)

    def compute_loss(self,X,y):
        pred = self.model(X)
        loss = self.loss_function(pred,y)
        return pred, loss

    def train_loop(self, 
                   train_dataloader,          # dataloader
                   update_every_n_batches=10, # how often to report 
                   epochs=1                   # how many epochs to go thru
                   ):
        """
        This function performs the training loop for the model.
        """      
        self.model.to(self.device)
        size = len(train_dataloader.dataset)
        num_batches = len(train_dataloader)
        step = 0
        for epoch in range(epochs):
            self.model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(self.device), y.to(self.device) 
                step += 1
                pred, loss = self.compute_loss(X, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self.report_scores(y_true=y, y_pred=pred,
                                    prefix='training',
                                    loss=loss.item(),
                                    epoch_num=(epoch+1),
                                    current_in_batch=(batch+1)*len(X),
                                    total_in_batch=size,
                                    step=step)
        return self.scores    
    
    def test_loop(self, 
                  test_dataloader,           # dataloader
                  update_every_n_batches=10  # how often to report
                  ):
        self.model.to(self.device)
                
        # one epoch pass
        self.model.eval()
        step = 0
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                step += 1
                pred, loss = self.compute_loss(X, y)
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self.report_scores(y_true=y, y_pred=pred,
                                    prefix='testing',
                                    loss=loss.item(),
                                    epoch_num=1,
                                    current_in_batch=(batch+1)*len(X),
                                    total_in_batch=size,
                                    step=step)
                
                
    def train(self, 
                   train_dataloader,          # dataloader
                   test_dataloader,
                   update_every_n_batches=10, # how often to report 
                   epochs=1,                   # how many epochs to go thru
                   patience=2, 
                   min_delta=0.1,
                   ):    
        self.model.to(self.device)
        train_size = len(train_dataloader.dataset)
        test_size = len(test_dataloader.dataset)
        num_batches = len(train_dataloader)
        step = 0
        best_loss = None
        waiting = 0
        for epoch in range(epochs):
            self.model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(self.device), y.to(self.device) 
                step += 1
                pred, train_loss = self.compute_loss(X, y)
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self.report_scores(y_true=y, y_pred=pred,
                                    prefix='training',
                                    loss=train_loss.item(),
                                    epoch_num=(epoch+1),
                                    current_in_batch=(batch+1)*len(X),
                                    total_in_batch=train_size,
                                    step=step)
            with torch.no_grad():
                for batch, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    step += 1
                    pred, val_loss = self.compute_loss(X, y)
                    if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                        self.report_scores(y_true=y, y_pred=pred,
                                        prefix='testing',
                                        loss=val_loss.item(),
                                        epoch_num=(epoch+1),
                                        current_in_batch=(batch+1)*len(X),
                                        total_in_batch=test_size,
                                        step=step)
            # Implement early stopping here
            if best_loss is None:
                best_loss = val_loss.item()
            elif best_loss - val_loss.item() > min_delta:
                best_loss = val_loss.item()
                waiting = 0
            else:
                waiting += 1
                if waiting >= patience:
                    print("Early stopping!")
                    break
        return self.scores  
    
    
    
    
    # def train(self, 
    #             train_dataloader,          # train dataloader
    #             test_dataloader,           # test data loader
    #             update_every_n_batches=10, # how often to report 
    #             epochs=1                   # how many epochs to go thru
    #             patience=5, 
    #             min_delta=0.001,
    #             ):
    #     """
    #     This function performs the training loop for the model with early stopping if possible
    #     """
    #     self.model.to(self.device)
    #     def compute_loss(X, y):
    #         pred = self.model(X)
    #         loss = self.loss_function(pred, y)
    #         loss.backward()
    #         return pred, loss

    #     def update_model():
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     def report(batch, X, pred, loss, step):
    #         if batch % update_every_n_batches == 0 or batch == num_batches - 1:
    #             self.report_scores(y_true=y, y_pred=pred,
    #                             prefix='training',
    #                             loss=loss.item(),
    #                             epoch_num=(epoch+1),
    #                             current_in_batch=(batch+1)*len(X),
    #                             total_in_batch=size,
    #                             step=step)

    #     size = len(train_dataloader.dataset)
    #     num_batches = len(train_dataloader)
    #     step = 0

    #     for epoch in range(epochs):
    #         self.model.train()
    #         for batch, (X, y) in enumerate(train_dataloader):
    #             X, y = X.to(self.device), y.to(self.device) 
    #             step += 1
    #             pred, loss = compute_loss(X, y)
    #             update_model()
    #             report(batch, X, pred, loss,step)
    #     return self.scores   