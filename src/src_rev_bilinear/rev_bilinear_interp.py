import torch
from torch.autograd import gradcheck
import torch.nn.functional as F
import rev_bilinear

import numpy as np

class RevBilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pc, pc_value, grid_size):
        pc_grid_index = rev_bilinear.cal_pc_grid_index(pc, grid_size)
        weight_sum = rev_bilinear.cal_weight_sum(pc, pc_grid_index, grid_size)
        torch.cuda.synchronize()
        weight_sum[weight_sum == 0] = 1
        grid_value = rev_bilinear.cal_grid_value(pc, pc_value, pc_grid_index, grid_size)
        torch.cuda.synchronize()
        grid_value = grid_value / weight_sum.unsqueeze(1)
        ctx.save_for_backward(pc, pc_value, grid_value, weight_sum, pc_grid_index)
        ctx.grid_size = grid_size
        return grid_value

    @staticmethod
    def backward(ctx, grad_grid_value):
        pc, pc_value, grid_value, weight_sum, pc_grid_index = ctx.saved_tensors
        grad_pc = grad_pc_value = None

        if ctx.needs_input_grad[1]:
            grad_pc_value = rev_bilinear.cal_pc_value_grad(grad_grid_value, pc, weight_sum, pc_grid_index, ctx.grid_size)
            torch.cuda.synchronize()
        if ctx.needs_input_grad[0]:
            grad_pc = rev_bilinear.cal_pc_grad(grad_grid_value, grid_value, pc, weight_sum, pc_value, pc_grid_index, ctx.grid_size)
            torch.cuda.synchronize()
            
        import pdb;pdb.set_trace()        

        return grad_pc, grad_pc_value, None

if __name__ == '__main__':
    #pc0 = torch.rand(1, 2, 6, dtype=torch.float32, requires_grad=True).cuda()
    #pc_value0 = torch.rand(1, 1, 6, dtype=torch.float32, requires_grad=True).cuda()
    pc = torch.tensor([[0.1, 0.3, 0.1, 0.6, 0.1, 0.9],
                       [0.1, 0.1, 0.3, 0.1, 0.6, 0.9]]).cuda()
    pc = pc[None,:,:]
    pc_value = torch.tensor([0,0.1,0.1,0.4,0.4,1.0]).cuda()
    pc_value = pc_value[None,None,:]
    
    pc.requires_grad_(True)
    pc_value.requires_grad_(True)

    grid_value = RevBilinear.apply(pc, pc_value, 4)

    input = (pc, pc_value, 4)
    test = gradcheck(RevBilinear.apply, input, eps=1e-3, atol=1e-3)
    #print(test)
