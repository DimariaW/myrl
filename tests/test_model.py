from myrl.model import Model
import torch


class TestModel(Model):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)


device = torch.device("cuda:0")
test_model1 = TestModel().to(device)
test_model2 = TestModel()


weights = test_model1.get_weights()

test_model1.set_weights(weights)

test_model1.sync_weights_to(test_model2)

test_model2.sync_weights_to(test_model1)
