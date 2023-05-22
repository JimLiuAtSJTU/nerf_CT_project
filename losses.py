import torch
from torch import nn

class RGB_MSELoss(nn.Module):
    def __init__(self):
        super(RGB_MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss

class CT_MSELoss(nn.Module):
    def __init__(self):
        super(CT_MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        #print(targets.shape) # H*W, 3 or H*W, 1
        if targets.shape[1]>1:
            # in rgb format
            targets=targets[:,0]
        loss = self.loss(inputs['transmittance_coarse'], targets)
      #  loss += inputs['sigma_norm_coarse'].mean()*0.01
        if 'transmittance_fine' in inputs:
            loss += self.loss(inputs['transmittance_fine'], targets)

        return loss

               

loss_dict = {'mse': RGB_MSELoss,
             'ct':CT_MSELoss}