
import torch
import torch.nn as nn
from collections import OrderedDict


class Model(nn.Module):
    def __init___(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass

    def sync_weights_to(self, target_model: "Model", decay=0.0):
        """
        target model and current model must in same device

        :param target_model:
        :param decay:
        :return:
        """
        target_vars = target_model.state_dict()
        for name, var in self.state_dict().items():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)

    def get_weights(self):
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        new_weights = OrderedDict()
        for key in weights.keys():
            new_weights[key] = torch.from_numpy(weights[key])
        self.load_state_dict(new_weights)
