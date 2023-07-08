import torch
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_available_devices():
    """
    Return the list of available devices (CPUs, CUDA GPUs, or Metal Performance Shaders) on the current platform.

    This function checks if CUDA or Metal Performance Shaders (MPS) is available and returns the list of available 
    devices. It first checks if CUDA is available, and if it is, adds all the CUDA devices to the list. Next, it checks 
    if the MPS backend is available and, if it is, adds the MPS device to the list. As of the latest knowledge cutoff in 
    September 2021, PyTorch does not support device indexing for MPS, so we cannot distinguish between different MPS 
    devices. If no GPU devices are found (either CUDA or MPS), the function adds the CPU device to the list of devices.

    Returns:
        List[torch.device]: List of available devices. Each device is represented as a `torch.device` object.

    Example:
        >>> devices = get_available_devices()
        >>> for device in devices:
        >>>     print(device)
    """    
    devices = []

    # Check CUDA availability
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f'cuda:{i}'))

    # Check MPS (Metal Performance Shaders) availability
    # if 'torch.backends.mps' in dir():
    if torch.backends.mps.is_available():
        # MPS does not support device indexing as of my knowledge cutoff in September 2021.
        # So, just add the 'mps' device without index.
        devices.append(torch.device('mps'))

    # If no GPUs are found, add the CPU
    if len(devices) == 0:
        devices.append(torch.device('cpu'))
    return devices

class NNTrainer:
    
    SUPPORTED_DEVICES = {'cpu','cuda','mps'}
    
    def __init__(self,device = 'cpu',
                 model = None,
                 loss_function = None,
                 optimizer = None,
                 scoring_functions = {},
                 verbose_level = 1,
                 log_dir='_tb_log_dir'):
        self.device = device
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scoring_functions = scoring_functions
        self.scores = defaultdict(list)
        self.verbose_level = verbose_level
        self.writer = SummaryWriter(log_dir)

    def with_device(self, device):
        """
        Sets the device for the current instance.

        This function sets the device attribute of the current instance to the requested device. 
        If the requested device is 'cuda' and CUDA is available, it will set the device to 'cuda'. 
        If the requested device is 'mps' and the MPS backend is available, it will set the device to 'mps'. 
        In all other cases, or if the requested device is not available, the device will be set to 'cpu'.
        
        Args:
            device (str): The requested device. Valid values are 'cpu', 'cuda', and 'mps'.

        Example:
            >>> trainer = NNTrainer() 
            >>> trainer.with_device('cuda')
        """
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
    

    def with_loss_function(self, loss_function):
        """
        Sets the loss function for the current instance and returns the instance.

        Args:
            loss_function (torch.nn.Module): The loss function to be used.

        Returns:
            self: The instance of the class.

        Example:
            >>> trainer = NNTrainer()
            >>> trainer.with_loss_function(torch.nn.CrossEntropyLoss())
        """
        self.loss_function = loss_function
        return self
    
    def with_optimizer(self, optimizer):
        """
        Sets the optimizer for the current instance and returns the instance.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be used.

        Returns:
            self: The instance of the class.

        Example:
            >>> trainer = NNTrainer()
            >>> trainer.with_optimizer(torch.optim.SGD(model.parameters(), lr=0.01))
            >>> trainer.with_optimizer(torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001, betas=(0.9, 0.99), eps=1e-8))
        """
        self.optimizer = optimizer
        return self
    
    def with_model(self, model):
        """
        Sets the model for the current instance and returns the instance.

        Args:
            model (torch.nn.Module): The model to be used

        Returns:
            self: The instance of the class.

        Example:
            >>> trainer = NNTrainer() 
            >>> trainer.with_model(MyModel())
        """
        self.model = model
        return self
    
    def with_scoring_functions(self, scoring_functions):
        """
        Sets the scoring functions for the current instance and returns the instance.

        Args:
            scoring_functions (dict): The scoring functions to be used.

        Returns:
            self: The instance of the class.

        Example:
            >>> trainer = NNTrainer()
            >>> trainer.with_scoring_functions({'accuracy':score_accuracy})
        """
        self.scoring_functions = scoring_functions
        return self
    
    def with_verbose_level(self, verbose_level):
        """
        Sets the verbosity level for the current trainer and returns the instance.

        Args:
            console_verbose_level (int): The level of verbosity for console output. 
                This could be implemented as follows:
                - 0: No output
                - 1: Basic output
                - 2: Detailed output

        Returns:
            self: The instance of the class.

        Example:
            >>> trainer = NNTrainer()
            >>> trainer.with_console_verbose(2)
        """
        self.verbose_level = verbose_level
        return self
    
    def _report_scores (self, 
                      y_true,            # ground truth 
                      y_pred,            # prediction
                      prefix,            # label/namespace
                      loss,              # loss
                      epoch_num,         # which epoch we're on
                      current_in_batch,  # how many observations processed
                      total_in_batch,    # how many observations in whole batch
                      step):             # which step of training/evaluation we're on
        """
        Internal function, subject to change
        This function reports the scores of a neural network training.
        """
        def report_to_tensorboard(name, value, step):
            if self.verbose_level > 1:
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

        if self.verbose_level > 0:
            print(header + body)

    def _compute_loss(self,X,y):
        """
        Internal function, subject to change
        """
        pred = self.model(X)
        loss = self.loss_function(pred,y)
        return pred, loss

    def train_loop(self, 
                   train_dataloader,          # dataloader
                   update_every_n_batches=10, # how often to report 
                   epochs=1                   # how many epochs to go thru
                   ):
        """
        Performs the training loop for the model, using only training data

        The function loops over the given number of epochs, and for each epoch, it loops over all batches in the train
        dataloader. For each batch, it computes the loss, performs backpropagation, and updates the model parameters. 
        Reporting (e.g., printing training loss, accuracy) is done every `update_every_n_batches` batches or at the end 
        of each epoch.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader that provides the training data in batches.
            update_every_n_batches (int, optional): How often to report training progress, in number of batches. Default 
                is 10.
            epochs (int, optional): The number of epochs to go through the training data. Default is 1.

        Returns:
            dict: The scores of the model on the training data after the last epoch. The contents of the dictionary 
                depend on the implementation of the `_report_scores` method.

        Example:
            >>> trainer = NNTrainer()  # assuming this class contains the train_loop method
            >>> scores = trainer.train_loop(train_dataloader, update_every_n_batches=20, epochs=5)
            >>> print(scores)
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
                pred, loss = self._compute_loss(X, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self._report_scores(y_true=y, y_pred=pred,
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
        
        """
        Performs a testing (evaluation) loop on the model.

        The function loops over all batches in the test dataloader, computes the loss, and reports the scores (e.g., 
        testing loss, accuracy) every `update_every_n_batches` batches or at the end of the testing.

        Args:
            test_dataloader (torch.utils.data.DataLoader): The dataloader that provides the testing data in batches.
            update_every_n_batches (int, optional): How often to report testing progress, in number of batches. Default 
                is 10.

        Returns:
            None. The scores are printed but not returned. To make this function return the scores, you need to modify 
            the function to store the scores in an instance variable or return them directly.

        Example:
            >>> tester = NNTrainer()  # assuming this class contains the test_loop method
            >>> tester.test_loop(test_dataloader, update_every_n_batches=20)
        """        
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
                pred, loss = self._compute_loss(X, y)
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self._report_scores(y_true=y, y_pred=pred,
                                    prefix='testing',
                                    loss=loss.item(),
                                    epoch_num=1,
                                    current_in_batch=(batch+1)*len(X),
                                    total_in_batch=size,
                                    step=step)
                
                
    def train_test_loop (self, 
                   train_dataloader,          # dataloader
                   test_dataloader,
                   update_every_n_batches=10, # how often to report 
                   epochs=1,                   # how many epochs to go thru
                   patience=2, 
                   min_delta=0.1,
                   ):  
        """
        Performs a training and testing loop on the model with early stopping.

        The function loops over the given number of epochs. For each epoch, it loops over all batches in the 
        train_dataloader and performs a training step. Then, it loops over all batches in the test_dataloader 
        and performs a validation step. The validation loss is compared to the best validation loss seen so far, 
        and early stopping is implemented based on the `patience` and `min_delta` parameters.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader that provides the training data in batches.
            test_dataloader (torch.utils.data.DataLoader): The dataloader that provides the testing data in batches.
            update_every_n_batches (int, optional): How often to report progress, in number of batches. Default is 10.
            epochs (int, optional): The number of epochs to go through the training data. Default is 1.
            patience (int, optional): The number of epochs to wait before stopping if the validation loss does not 
                decrease by at least `min_delta`. Default is 2.
            min_delta (float, optional): The minimum decrease in validation loss to qualify as an improvement. Default 
                is 0.1.

        Returns:
            dict: The scores of the model on the training data after the last epoch. The contents of the dictionary 
                depend on the implementation of the `report_scores` method.

        Example:
            >>> trainer = NNTrainer()  # assuming this class contains the train_test_loop method
            >>> scores = trainer.train_test_loop(train_dataloader, test_dataloader, update_every_n_batches=20, epochs=5, 
            patience=3, min_delta=0.05)
            >>> print(scores)
        """
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
                pred, train_loss = self._compute_loss(X, y)
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                    self._report_scores(y_true=y, y_pred=pred,
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
                    pred, val_loss = self._compute_loss(X, y)
                    if batch % update_every_n_batches == 0 or batch == num_batches - 1:
                        self._report_scores(y_true=y, y_pred=pred,
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