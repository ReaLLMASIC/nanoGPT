import math
import torch
import torch.nn as nn
from torch.autograd import Function
import pdb

class BinaryLinearFunction(Function):
    
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # binarize weights by sign function
        weight_mask = (weight > 1) | (weight < -1)
        weight = torch.sign(weight)
        # save for grad computing
        ctx.save_for_backward(input, weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value larger than 1, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class BinarizedLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.binarized_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("b_weight", torch.Tensor(out_features, in_features))

        if bias:
            self.binarization_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.binarized_weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.binarization_bias is not None:
            self.binarization_bias.data.zero_()

    def forward(self, input):
        if self.binarization_bias is not None:
            self.b_weight = torch.sign(self.binarized_weight)
            return BinaryLinearFunction.apply(input, self.binarized_weight, self.binarization_bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"

    
class IRLinearFunction(Function):
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, t=None):
        # normalize weights
        weight_mean = torch.mean(weight)
        weight_std = torch.std(weight)
        weight_norm = (weight - weight_mean)/weight_std
        
        # compute control variable k
        k = torch.max(torch.Tensor([1/t,1]))
        
        # binarize by EDE function
        weight_b = k * torch.tanh(t * weight_norm)
        
        # save for grad computing
        ctx.save_for_backward(input, weight_b, weight_norm, bias, weight_std, t, k)
        
        # linear layer
        output = input.matmul(weight_b.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight_b, weight_norm, bias, weight_std, t, k = ctx.saved_variables
        
#         pdb.set_trace()
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_b)

        if ctx.needs_input_grad[1]:
            grad_weight_b = grad_output.transpose(-1, -2).matmul(input)
            grad_binary = k * t * (1 - torch.square(torch.tanh(t * weight_norm)))
            grad_weight = grad_weight_b * grad_binary 

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None
    

class IRLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, bias=True):
        super(IRLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.t = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.t = None
        self.weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return IRLinearFunction.apply(input, self.weight, self.bias, self.t)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"
    

def binarizer(model, binarize_layer='basic', qk_only=False, qv_only=False, kv_only=False):
    """
    Recursively replace linear layers with binary layers
    ---------
    Arguments:
    model      - Model to be binarized
    pattern    - Binarization pattern
    skip_final - whether to leave final layer unbinarized
    qk_only    - whether to only binarize Q net and K net, leaving V net
    ---------
    No return, the original model is binarized
    """
    
    for i, block in enumerate(model.transformer.h):
        for sub_name, sub_layer in getattr(block, "mlp").named_children():
            if type(sub_layer) == nn.Linear:
            # Binarization
                if (kv_only == True) & (sub_name == 'c_attn_q'):
                    continue
                if (qv_only == True) & (sub_name == 'c_attn_k'):
                    continue
                if (qk_only == True) & (sub_name == 'c_attn_v'):
                    continue
                    
                if binarize_layer == 'basic':
                    b = BinarizedLinear(sub_layer.in_features, sub_layer.out_features)
                elif binarize_layer == 'ir':
                    b = IRLinear(sub_layer.in_features, sub_layer.out_features)  
                model.__dict__["_modules"]["transformer"]["h"][i].mlp[sub_name] = b
        for sub_name, sub_layer in getattr(block, "attn").named_children():
            if type(sub_layer) == nn.Linear:
                if (kv_only == True) & (sub_name == 'c_attn_q'):
                    continue
                if (qv_only == True) & (sub_name == 'c_attn_k'):
                    continue
                if (qk_only == True) & (sub_name == 'c_attn_v'):
                    continue
                if binarize_layer == 'basic':
                    b = BinarizedLinear(sub_layer.in_features, sub_layer.out_features)
                elif binarize_layer == 'ir':
                    b = IRLinear(sub_layer.in_features, sub_layer.out_features)  
                model.__dict__["_modules"]["transformer"]["h"][i].attn[sub_name] = b