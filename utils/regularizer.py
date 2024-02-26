
import torch
import numpy as np

class Regularizer():
    def __init__(self, coef, device):
        self.device = device
        self.coef = coef

    def cal_targets(self, hnet, task_id):
        ret = []
        with torch.no_grad():
            for id in range(task_id):
                 id = torch.Tensor([id]).long().to(self.device)
                 ret.append([p.detach() for p in hnet.forward(id)])
        return ret
    
    def cal_reg(self, hnet, task_id, targets):
        reg = 0
        for id in range(task_id):
            id = torch.Tensor([id]).long().to(self.device)
            delta_weights_predicted = hnet.forward(id)
            target = targets[id]
            W_target = torch.cat([w.view(-1) for w in target])
            W_predicted = torch.cat([w.view(-1) for w in delta_weights_predicted])
            reg_i = (W_target - W_predicted).pow(2).sum()
            reg += reg_i
        
        return self.coef * reg / task_id
    
        
