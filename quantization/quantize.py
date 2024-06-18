import torch
from torch import nn
from torch.nn import functional as F


def quantize(tensor, bits):
    """
    Quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: Quantized code
    """

    # Steps:
    # Normalizes the tensor values to the range [0,𝑠]
    # Uses stochastic rounding to determine the quantized values.
    # Combines the quantized values with the original signs.
    # Returns the scaling factor and the quantized tensor.

    # maximum integer value that can be represented with the given number of bits. For example, if bits=8, s=255 (2^8-1)
    s = (1 << bits) - 1

    # norm = torch.norm(tensor)
    norm = tensor.abs().max()

    # captures the sign of each element in the tensor
    sign_array = torch.sign(tensor).to(dtype=torch.int8)

    # scales the absolute values of the tensor to the range [0,𝑠]
    l_array = torch.abs(tensor) / norm * s
    l_array_floored = l_array.to(dtype=torch.int)

    prob_array = l_array - l_array_floored
    # fractional part of l_array, clamped between 0 and 1 (rescaled so min is 0 and max is 1)
    prob_array = torch.clamp(prob_array, min=0.0, max=1.0)


    # stochastic rounding: draw 0 or 1s from a Bernoulli distribution with probability equal to the corresponding element
    mask = torch.bernoulli(prob_array)

    # final quantized array. Elements are incremented by 1 if the corresponding element in mask is 1 (stochastic rounding)
    xi_array = l_array_floored + mask
    xi_array = xi_array.to(dtype=torch.int32)

    # combines the sign and the quantized magnitude to get the final quantized tensor with the same sign as the original tensor
    sign_xi_array = (sign_array * xi_array).to(dtype=torch.int8)
    norm = norm / s

    return norm, sign_xi_array


def dequantize(norm, sign_xi_array):
    """
    Dequantize the quantization code
    :param norm: Norm of code
    :param sign_xi_array: Rounded vector of code
    :return: Dequantized weights
    """

    # weight ≈ (norm / s) * (tensor / norm * s)
    weights = norm * sign_xi_array

    return weights


class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    @staticmethod
    def forward(ctx, input, bits=7):
        """
        Forward pass
        :param ctx: Context object to store information for the backward pass (not used in this case)
        :param input: The input tensor to be quantized
        :param bits: The number of bits for quantization (default is 7)
        :return: Dequantized tensor
        """
        # steps:
        # Quantize the input tensor using the quantize function.
        # Dequantize the quantized values using the dequantize function.
        # Return the dequantized tensor, which approximates the input tensor but includes the quantization error.
        norm, quantized_weight = quantize(input, bits)
        return dequantize(norm, quantized_weight)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE): passes grad_output through as the gradient with respect to the input
        # gradient is approximated by simply passing the gradient from the output directly to the input, 
        # ignoring the quantization operation
        return grad_output, None, None


_fake_quantize = FakeLinearQuantizationFunction.apply


class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, weight_bits, in_features, out_features, warmup_step=0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        self.weight_bits = weight_bits

        if self.weight_bits < 1:
            raise ValueError(f"weight_bits={self.weight_bits} must be higher than 0 ")
        
        self.warmup_step = warmup_step
        self.accumulation_bits = 32

        # Placeholder for quantized weights during training
        self._fake_quantized_weight = None
        if kwargs.get("bias", True):
            self.register_buffer("quantized_bias", None)
            self.register_buffer("bias_norm", None)

        self.register_buffer("_step", torch.zeros(1))

        self.register_buffer("quantized_weight", None)
        self.register_buffer("weight_norm", None)

    def training_quantized_forward(self, input):
        """Fake quantizes weights. Function should only be used while training"""
        assert self.training, "Should be called only during training"

        # Applies the fake quantization to the weights
        self._fake_quantized_weight = _fake_quantize(self.weight, self.weight_bits)
        # Uses the quantized weights to compute the output using F.linear
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. Function should be called only during inference"""
        assert not self.training, "Should be called only during inference"

        # Compute the dequantized weight
        weight = self.weight_norm * self.quantized_weight

        # Compute the dequantized bias
        if self.bias is not None:
            bias = self.bias_norm * self.quantized_bias

        # Uses the dequantized weights and bias to compute the output using F.linear
        if self.bias:
            out = F.linear(input, weight, bias)
        else:
            out = F.linear(input, weight)

        return out

    def _eval(self):
        """Sets the model for inference by quantizing the model"""
        self.weight_norm, self.quantized_weight = quantize(self.weight, self.weight_bits)

        if self.bias is not None:
            self.bias_norm, self.quantized_bias = quantize(self.bias, self.accumulation_bits)

    def forward(self, input):
        """Passes the input through the model during training and inference"""
        if self.training:
            if self._step > self.warmup_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            # Prepares the model for inference by quantizing weights and bias
            self._eval()
            # Uses quantized weights and bias to compute the output
            out = self.inference_quantized_forward(input)
        return out