import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.nn.init as init

class HyperBase(nn.Module):
    def __init__(self, task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims):
        super().__init__()
        self.use_block_emb = use_block_emb
        self.block_nums = block_nums
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.block_emb_dims = block_emb_dims
        self.block_emb_sharing = block_emb_sharing
        self.input_dim = task_emb_dims

        self.task_embs = nn.Embedding(task_nums, task_emb_dims)
        init.normal_(self.task_embs.weight, mean=0, std=0.1)
        if use_block_emb:
            self.input_dim += block_emb_dims
            if block_emb_sharing:
                self.register_buffer("block_emb_input", torch.arange(0, self.block_nums))
                self.block_emb = nn.Embedding(self.block_nums, block_emb_dims)
                init.normal_(self.block_emb.weight, mean=0, std=0.1)                
            else:
                self.register_buffer("block_emb_input", torch.arange(0, self.block_nums * task_nums))
                self.block_emb = nn.Embedding(self.block_nums * task_nums, block_emb_dims)
                init.normal_(self.block_emb.weight, mean=0, std=0.1)

    def forward(self):
        pass

    @property
    def param_nums(self):
        nums = 0
        for param in self.parameters():
            nums += param.shape.numel()
        return nums

class HyperHead(nn.Module):
    def __init__(self, task_nums, task_emb_dims, out_dims):
        super().__init__()
        self.task_embs = nn.Embedding(task_nums, task_emb_dims)
        init.normal_(self.task_embs.weight, mean=0, std=0.1)
        
        self.out_dims = out_dims

        self.linears = nn.Sequential(
            nn.Linear(task_emb_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(100, out_dims[0] * out_dims[1])

    def forward(self, task_id):
        task_emb = self.task_embs(task_id)
        x = self.linears(task_emb)
        x = self.linear(x)
        x = x.view(self.out_dims[0], self.out_dims[1])
        return [x]
    
    @property
    def param_nums(self):
        nums = 0
        for param in self.parameters():
            nums += param.shape.numel()
        return nums
    
   
class HyperLoRA(HyperBase):
    def __init__(self, task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims, rank):
        super().__init__(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims)
        
        self.rank = rank
        if use_block_emb:
            self.linear_A = nn.Linear(100, self.in_dims * self.rank)
            self.linear_B = nn.Linear(100, self.out_dims * self.rank)
            
        else:
            self.linear_A = nn.Linear(100, self.in_dims * self.rank * self.block_nums)
            self.linear_B = nn.Linear(100, self.out_dims * self.rank * self.block_nums)

        self.linears = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )


    def forward(self, task_id):
        task_emb = self.task_embs(task_id)
        if self.use_block_emb:
            assert self.block_emb_dims > 0
            if self.block_emb_sharing:
                block_emb = self.block_emb(self.block_emb_input)
            else:
                start_idx = task_id.item() * self.block_nums
                end_idx = (task_id + 1) * self.block_nums
                block_emb = self.block_emb(self.block_emb_input[start_idx : end_idx])

            task_emb = task_emb.repeat(self.block_nums, 1)
            x = self.linears(torch.concat([task_emb, block_emb], dim=-1)) 
            lora_A = self.linear_A(x).view(-1, self.rank, self.in_dims)
            lora_B = self.linear_B(x).view(-1, self.out_dims, self.rank)
            rets = [lora for lora in torch.matmul(lora_B, lora_A)]
            return rets
            
        x = self.linears(task_emb)
        lora_A = self.linear_A(x).view(self.block_nums, self.rank, self.in_dims)
        lora_B = self.linear_B(x).view(self.block_nums, self.out_dims, self.rank)
        rets = [lora for lora in torch.matmul(lora_B, lora_A)]
        return rets
    
    @property
    def param_nums(self):
        nums = 0
        for param in self.parameters():
            nums += param.shape.numel()
        return nums
    
class HyperPrefix(HyperBase):
    def __init__(self, task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims, prefix_length):
        super().__init__(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims)
        
        self.prefix_length = prefix_length
        if use_block_emb:
            self.linear_prefixes = nn.Linear(100, 2 * self.in_dims * self.prefix_length)
            
        else:
            self.linear_prefixes = nn.Linear(100, 2 * self.in_dims * self.prefix_length * self.block_nums)

        self.linears = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

    def forward(self, task_id):
        task_emb = self.task_embs(task_id)
        if self.use_block_emb:
            assert self.block_emb_dims > 0
            if self.block_emb_sharing:

                block_emb = self.block_emb(self.block_emb_input)
            else:
                start_idx = task_id.item() * self.block_nums
                end_idx = (task_id + 1) * self.block_nums
                block_emb = self.block_emb(self.block_emb_input[start_idx : end_idx])

            task_emb = task_emb.repeat(self.block_nums, 1)
            x = self.linears(torch.concat([task_emb, block_emb], dim=-1)) 
            prefixes = self.linear_prefixes(x).view(-1, 2, self.prefix_length, self.in_dims)
           
            rets = [prefix for prefix in prefixes]
            return rets
            
        x = self.linears(task_emb)
        prefixes = self.linear_prefixes(x).view(self.block_nums, 2, self.prefix_length, self.in_dims)
        rets = [prefix for prefix in prefixes]
        return rets

class HyperAdapter(HyperBase):
    def __init__(self, task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims, down_sample_dim):
        super().__init__(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims)
        self.down_sample_dim = down_sample_dim
        if use_block_emb:
            self.linear_adapters = nn.Linear(100, 2 * self.in_dims * self.down_sample_dim)
            
        else:
            self.linear_adapters = nn.Linear(100, 2 * self.in_dims * self.down_sample_dim * self.block_nums)

        self.linears = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

    def forward(self, task_id):
        task_emb = self.task_embs(task_id)
        if self.use_block_emb:
            assert self.block_emb_dims > 0
            if self.block_emb_sharing:
                block_emb = self.block_emb(self.block_emb_input)
            else:
                start_idx = task_id.item() * self.block_nums
                end_idx = (task_id + 1) * self.block_nums
                block_emb = self.block_emb(self.block_emb_input[start_idx : end_idx])

            task_emb = task_emb.repeat(self.block_nums, 1)
            x = self.linears(torch.concat([task_emb, block_emb], dim=-1)) 
            adapters = self.linear_adapters(x).view(-1, 2, self.down_sample_dim, self.in_dims)
           
            rets = [adapter for adapter in adapters]
            return rets
            
        x = self.linears(task_emb)
        adapters = self.linear_adapters(x).view(self.block_nums, 2, self.down_sample_dim, self.in_dims)
        rets = [adapter for adapter in adapters]
        return rets
    
class HyperNetwork(nn.Module):
    def __init__(self, target_shapes, task_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, peft, **kwargs):
        super().__init__()
        self.peft = peft
        block_shape = target_shapes[0]
        head_shape = target_shapes[-1]

        if peft in ['lora', 'prefix', 'adapter']:
            in_dims = block_shape[1]
            out_dims = block_shape[0]
            block_nums = len(target_shapes) - 1

        if peft == 'lora':
            rank = kwargs['rank']
            self.hyper_peft = HyperLoRA(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims, rank)
        elif peft == 'prefix':
            prefix_length = kwargs['prefix_length']
            self.hyper_peft = HyperPrefix(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing, in_dims, out_dims, prefix_length)
        elif peft == 'adapter':
            down_sample_dims = kwargs['down_sample_dims']
            self.hyper_peft = HyperAdapter(task_nums, block_nums, task_emb_dims, block_emb_dims, use_block_emb, block_emb_sharing,  in_dims, out_dims, down_sample_dims)
        
        self.hyper_head = HyperHead(task_nums, task_emb_dims, head_shape)

    def forward(self, task_id):
        assert self.peft in ['lora', 'prefix', 'adapter']

        return self.hyper_peft(task_id) + self.hyper_head(task_id)
        
           
    @property
    def param_nums(self):
        nums = 0
        for param in self.parameters():
            nums += param.shape.numel()
        return nums

