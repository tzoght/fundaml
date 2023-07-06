from mltz_base.models import SampleNNClassifier
from mltz_base.trainers import NNTrainer
from mltz_base.scores import score_accuracy
import unittest
import torch
from torch import Tensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class TestTrainerBase(unittest.TestCase):

    def test_forward(self):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        learning_rate = 1e-3
        batch_size = 64
        epochs = 1
        weight_decay = 0.01
        
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)        
        
        model = SampleNNClassifier(short_name="Sample_NN_Classifier")
        loss_fn = nn.CrossEntropyLoss()
        trainer = NNTrainer()
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)

        trainer.with_model(model).with_optimizer(optimizer).with_loss_function(loss_fn)
        trainer.with_scoring_functions({'accuracy':score_accuracy}).with_device('cpu') # 'mps' or 'cuda'
        # scores = trainer.train_loop(train_dataloader,update_every_n_batches=20,epochs=epochs)
        # scores = trainer.test_loop(test_dataloader,update_every_n_batches=2)
        scores = trainer.train(train_dataloader, test_dataloader,update_every_n_batches=100, epochs=5)
        print(scores)
        
if __name__ == '__main__':
    unittest.main()