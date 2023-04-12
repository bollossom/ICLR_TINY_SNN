from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
# from . import surrogate, base
import math
# try:
#     import cupy
#     from . import neuron_kernel, cu_kernel_opt
# except ImportError:
#     neuron_kernel = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

tab4_str = '\t\t\t\t'  # used for aligning code
curly_bracket_l = '{'
curly_bracket_r = '}'


def heaviside(x: torch.Tensor):
    '''
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: 输入tensor
    :return: 输出tensor

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    * :ref:`中文API <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    '''
    return (x >= 0).to(x)


def check_manual_grad(primitive_function, spiking_function, eps=1e-5):
    '''
    :param primitive_function: 梯度替代函数的原函数
    :type primitive_function: callable
    :param spiking_function: 梯度替代函数
    :type spiking_function: callable
    :param eps: 最大误差
    :type eps: float

    梯度替代函数的反向传播一般是手写的，可以用此函数去检查手写梯度是否正确。

    此函数检查梯度替代函数spiking_function的反向传播，与原函数primitive_function的反向传播结果是否一致。“一致”被定义为，两者的误差不超过eps。

    示例代码：

    .. code-block:: python

        surrogate.check_manual_grad(surrogate.ATan.primitive_function, surrogate.atan.apply)
    '''
    alpha = torch.tensor(1.0, dtype=torch.float)
    x = torch.arange(-16, 16, 32 / 8192)
    x.requires_grad_(True)
    primitive_function(x, alpha).sum().backward()
    x_grad_auto = x.grad.clone()
    x.grad.zero_()
    spiking_function(x, alpha).sum().backward()
    x_grad_manual = x.grad.clone()
    assert (x_grad_manual - x_grad_auto).abs().max().item() <= eps, 'x.grad is wrong!'
    print('grad check pass')


class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)


class MultiArgsSurrogateFunctionBase(nn.Module):
    def __init__(self, spiking: bool, *args, **kwargs):
        super().__init__()
        self.spiking = spiking

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x_abs = ctx.saved_tensors[0].abs()
            mask = (x_abs > (1 / ctx.alpha))
            grad_x = (grad_output * (- (ctx.alpha ** 2) * x_abs + ctx.alpha)).masked_fill_(mask, 0)
        return grad_x, None


class PiecewiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseQuadratic.__init__-en>`
        .. _PiecewiseQuadratic.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段二次函数的梯度（三角形函数）的脉冲发放函数。反向传播为

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.*
            :width: 100%

        该函数在文章 [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_ 中使用。

        * :ref:`中文API <PiecewiseQuadratic.__init__-cn>`
        .. _PiecewiseQuadratic.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise quadratic surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.*
            :width: 100%

        The function is used in [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_.

        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1.5$')

    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1.5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise quadratic surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 * (- ctx.alpha * ctx.saved_tensors[0].abs()).exp_() * grad_output

        return grad_x, None


class PiecewiseExp(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseExp.__init__-en>`
        .. _PiecewiseExp.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段指数函数的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.*
            :width: 100%

        该函数在文章 [#SLAYER]_ [#neftci2019surrogate]_ 中使用。

        * :ref:`中文API <PiecewiseExp.__init__-cn>`
        .. _PiecewiseExp.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise exponential surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.*
            :width: 100%

        The function is used in [#SLAYER]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2 - 1
        exp_x = (mask_sign * x * -alpha).exp_() / 2

        return mask_nonnegative - exp_x * mask_sign

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise exponential surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None


class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        '''
        * :ref:`API in English <Sigmoid.__init__-en>`
        .. _Sigmoid.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用sigmoid的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        对应的原函数为

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.*
            :width: 100%

        该函数在文章 [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ 中使用。

        * :ref:`中文API <Sigmoid.__init__-cn>`
        .. _Sigmoid.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.*
            :width: 100%

        The function is used in  [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha};
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha);
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=5$')

    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Sigmoid surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / (2 * ctx.alpha * (1 / ctx.alpha + ctx.saved_tensors[0].abs()).pow_(2))
        return grad_x, None


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <SoftSign.__init__-en>`
        .. _SoftSign.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用soft sign的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}} = \\frac{1}{2\\alpha(\\frac{1}{\\alpha} + |x|)^{2}}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)
            = \\frac{1}{2} (\\frac{x}{\\frac{1}{\\alpha} + |x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/SoftSign.*
            :width: 100%

        该函数在文章 [#SuperSpike]_ [#neftci2019surrogate]_ 中使用。

        * :ref:`中文API <SoftSign.__init__-cn>`
        .. _SoftSign.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The soft sign surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/SoftSign.*
            :width: 100%

        The function is used in [#SuperSpike]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (F.softsign(x * alpha) + 1) / 2

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('SoftSign surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None


class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <ATan.__init__-en>`
        .. _ATan.__init__-cn:

        反向传播时使用反正切函数arc tangent的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.*
            :width: 100%

        * :ref:`中文API <ATan.__init__-cn>`
        .. _ATan.__init__-en:

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.*
            :width: 100%
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.ATan(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.ATan(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('ATan surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class nonzero_sign_log_abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / (1 / ctx.alpha + ctx.saved_tensors[0].abs())

        return grad_x, None


class NonzeroSignLogAbs(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <LogAbs.__init__-en>`
        .. _LogAbs.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        .. warning::
            原函数的输出范围并不是(0, 1)。它的优势是反向传播的计算量特别小。

        反向传播时使用NonzeroSignLogAbs的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        对应的原函数为

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        其中

            .. math::
                \\mathrm{NonzeroSign}(x) =
                \\begin{cases}
                1, & x \\geq 0 \\\\
                -1, & x < 0 \\\\
                \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        该函数在文章  中使用。

        * :ref:`中文API <LogAbs.__init__-cn>`
        .. _LogAbs.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        .. admonition:: Warning
            :class: warning

            The output range the primitive function is not (0, 1). The advantage of this function is that computation
            cost is small when backward.

        The NonzeroSignLogAbs surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        where

        .. math::
            \\mathrm{NonzeroSign}(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            -1, & x < 0 \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        The function is used in  .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        # the gradient of ``(heaviside(x) * 2 - 1) * (alpha * x.abs() + 1).log()`` by autograd is wrong at ``x==0``
        mask_p = heaviside(x) * 2 - 1
        return mask_p * (alpha * mask_p * x + 1).log()

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1$')

    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('NonzeroSignLogAbs surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (- (ctx.saved_tensors[0] * ctx.alpha).pow_(2)).exp_() * (
                        ctx.alpha / math.sqrt(math.pi))

        return grad_x, None


class Erf(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <Erf.__init__-en>`
        .. _Erf.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用高斯误差函数(erf)的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\pi}}e^{-\\alpha^2x^2}

        对应的原函数为

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ./_static/API/clock_driven/surrogate/Erf.*
            :width: 100%

        该函数在文章 [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_ 中使用。

        * :ref:`中文API <Erf.__init__-cn>`
        .. _Erf.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The Gaussian error (erf) surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\pi}}e^{-\\alpha^2x^2}

        The primitive function is defined by

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ./_static/API/clock_driven/surrogate/Erf.*
            :width: 100%

        The function is used in [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_.
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.erfc_(-alpha * x) / 2

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Gaussian error surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            mask_width = (ctx.saved_tensors[0].abs() < ctx.w)
            mask_c = mask_width.logical_not()
            grad_x = grad_output * ctx.saved_tensors[0].masked_fill(mask_width, 1 / ctx.w).masked_fill(mask_c, ctx.c)
        return grad_x, None, None


class PiecewiseLeakyReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, w=1., c=0.01, spiking=True):
        '''
        * :ref:`API in English <PiecewiseLeakyReLU.__init__-en>`
        .. _PiecewiseLeakyReLU.__init__-cn:

        :param w: ``-w <= x <= w`` 时反向传播的梯度为 ``1 / 2w``
        :param c: ``x > w`` 或 ``x < -w`` 时反向传播的梯度为 ``c``
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        分段线性的近似脉冲发放函数。梯度为

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_ 中使用。

        * :ref:`中文API <PiecewiseLeakyReLU.__init__-cn>`
        .. _PiecewiseLeakyReLU.__init__-en:

        :param w: when ``-w <= x <= w`` the gradient is ``1 / 2w``
        :param c: when ``x > w`` or ``x < -w`` the gradient is ``c``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_.
        '''
        super().__init__(spiking)
        assert w > 0.
        self.w = w
        self.c = c
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.w, self.c)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def primitive_function(x: torch.Tensor, w, c):
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (- cw + 1)) \
                   + mask2 * (x / (2 * w) + 1 / 2)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > {w})
            {curly_bracket_l}
                {y} = {c};
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = {w_inv};
            {curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn({w}));
            {tab4_str}half2 {y} = __hadd2(__hmul2(__float2half2_rn({c}),  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn({w_inv})));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $w=1, c=0.1$')

    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $w=1, c=0.1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('PiecewiseLeakyReLU surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class squarewave_fourier_series(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n: int, T_period: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.n = n
            ctx.T_period = T_period
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.
        x = ctx.saved_tensors[0]
        w = math.pi * 2. / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.) * w * x)

        grad_x *= 4. / ctx.T_period
        grad_x *= grad_output

        return grad_x, None, None


class SquarewaveFourierSeries(MultiArgsSurrogateFunctionBase):
    def __init__(self, n: int = 2, T_period: float = 8, spiking=True):
        super().__init__(spiking)
        assert isinstance(n, int) and T_period > 0.
        self.n = n
        self.T_period = T_period
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.n, self.T_period)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return squarewave_fourier_series.apply(x, w, c)

    @staticmethod
    def primitive_function(x: torch.Tensor, n: int, T_period: float):
        w = math.pi * 2. / T_period
        ret = torch.zeros_like(x.data)
        for i in range(1, n):
            c = (2 * i - 1.)
            ret += torch.sin(c * w * x) / c

        return 0.5 + 2. / math.pi * ret

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            raise NotImplementedError
        elif dtype == 'fp16':
            raise NotImplementedError
        else:
            raise NotImplementedError

        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # import torch
    # from spikingjelly.clock_driven import surrogate
    # from matplotlib import pyplot as plt
    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    #
    # c_list = []
    # for n in [2, 4, 8]:
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=False)
    #     y = surrogate_function(x)
    #     plt.plot(x.data, y.data, label=f'Primitive, $n={n}$')
    #     c_list.append(plt.gca().lines[-1].get_color())
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/clock_driven/surrogate/SquarewaveFourierSeries1.pdf')
    # plt.savefig('./docs/source/_static/API/clock_driven/surrogate/SquarewaveFourierSeries1.svg')
    # plt.clf()
    # for i, n in enumerate([2, 4, 8]):
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=True)
    #     x = x.detach()
    #     x.requires_grad_(True)
    #     y = surrogate_function(x)
    #     z = y.sum()
    #     z.backward()
    #     plt.plot(x.data, x.grad, label=f'Gradient, $n={n}$', c=c_list[i])
    #     x.grad.zero_()
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/clock_driven/surrogate/SquarewaveFourierSeries2.pdf')
    # plt.savefig('./docs/source/_static/API/clock_driven/surrogate/SquarewaveFourierSeries2.svg')


import torch
import torch.nn as nn
import copy


class MemoryModule(nn.Module):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _memoriesModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _memoriesModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _memoriesModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _memoriesModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a spiking neuron). The reset value of this variable will be ``value``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _memoriesModule.reset-cn:

        重置所有有状态变量。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _memoriesModule.reset-en:

        Reset all stateful variables.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _memoriesModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _memoriesModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)

        for key, value in self._memories_rv.items():
            if isinstance(value, torch.Tensor):
                self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class BaseNode(MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)
        self.v_threshold = v_threshold
        # self.v_threshold = torch.nn.Parameter(torch.as_tensor(v_threshold))
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike


class AdaptBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 v_rest: float = 0., w_rest: float = 0, tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super.__init__(v_threshold, v_reset, v_rest, surrogate_function, detach_reset)

        self.register_memory('w', w_rest)

        self.w_rest = w_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    def neuronal_adaptation(self):
        self.w = self.w + 1. / self.tau_w * (self.a * (self.v - self.v_rest) - self.w) + self.b * self.spike

    def extra_repr(self):
        return super.extra_repr + f', w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_adaptation()
        self.neuronal_reset()
        return self.spike


class AdaptLIFNode(AdaptBaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0.,
                 b: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, v_rest, w_rest, tau_w, a, b, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_rest == 0.:
                self.v = self.v + (x - self.v - self.w) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_rest) - self.w) / self.tau

        else:
            if self.v_rest == 0.:
                self.v = self.v - (self.v + self.w) / self.tau + x
            else:
                self.v = self.v - ((self.v - self.v_rest) + self.w) / self.tau + x


class Izhikevich_test(BaseNode):
    def __init__(self, tau=2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False, a: float = 0.002,
                 b: float = 0.020, d: float = 0.2, k: float = 1., vr: float = -0.05, vt: float = 1.):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.a = a
        self.b = b
        self.d = d
        self.vr = vr
        self.vt = vt
        self.k = k
        self.tau = tau
        self.register_memory('u', b * vr)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (self.k * (self.v - self.vr) * (self.v - self.vt) - self.u + x) / self.tau
        self.u = self.u + self.a * (self.b * (self.v - self.vr) - self.u)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        # hard reset
        self.v = (1. - spike) * self.v + spike * self.v_reset
        self.u = (1. - spike) * self.u + spike * (self.u + self.d)


class HHNode(BaseNode):
    def __init__(self, tau=2., v_threshold: float = 1., v_reset: float = 0., y_init=0.317676914060697,
                 m_init=0.052932485257250, h_init=0.596120753508460,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False, V_Na: float = 115.,
                 V_K: float = -12., V_L: float = 10.6, gbar_Na: float = 120., gbar_K: float = 36, gbar_L: float = 0.3):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.V_Na = V_Na
        self.V_K = V_K
        self.V_L = V_L

        self.gbar_Na = gbar_Na
        self.gbar_K = gbar_K
        self.gbar_L = gbar_L
        self.tau = tau
        self.register_memory('y', y_init)
        self.register_memory('m', m_init)
        self.register_memory('h', h_init)

    def neuronal_charge(self, x: torch.Tensor):
        a_n = 0.01 * (10 - self.v) / (torch.exp((torch.as_tensor(10.) - self.v) / 10) - 1)
        b_n = 0.125 * torch.exp(-self.v / torch.as_tensor(80))
        a_m = 0.1 * (25 - self.v) / (torch.exp((torch.as_tensor(25) - self.v) / 10) - 1)
        b_m = 4 * torch.exp(-self.v / torch.as_tensor(18))
        b_h = 1 / (torch.exp(torch.as_tensor((30. - self.v) / 10)))
        a_h = 0.07 * torch.exp(-self.v / torch.as_tensor(20))

        g_Na = torch.as_tensor(self.gbar_Na * self.h * self.m ** 3)
        g_K = torch.as_tensor(self.gbar_K * (self.y ** 4))
        I_Na = torch.as_tensor(g_Na * (self.v - self.V_Na))
        I_K = torch.as_tensor(g_K * (self.v - self.V_K))
        I_L = torch.as_tensor(self.gbar_L * (self.v - self.V_L))
        self.v = self.v + (x - I_Na.detach() - I_K.detach() - I_L.detach()) / self.tau
        self.y = self.y + (a_n * (1 - self.y) - b_n * self.y) / self.tau
        self.m = self.m + (a_m * (1 - self.m) - b_m * self.m) / self.tau
        self.h = self.h + (a_h * (1 - self.h) - b_h * self.h) / self.tau


class PHHNode(BaseNode):
    def __init__(self, tau=2., v_threshold: float = 1., v_reset: float = 0., y_init=0.317676914060697,
                 m_init=0.052932485257250, h_init=0.596120753508460,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False, V_Na: float = 115.,
                 V_K: float = -12., V_L: float = 10.6, gbar_Na: float = 120., gbar_K: float = 36, gbar_L: float = 0.3):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.V_Na = V_Na
        self.V_K = V_K
        self.V_L = V_L

        self.gbar_Na = gbar_Na
        self.gbar_K = gbar_K
        self.gbar_L = gbar_L
        self.register_memory('y', y_init)
        self.register_memory('m', m_init)
        self.register_memory('h', h_init)
        init_w = - math.log(tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def neuronal_charge(self, x: torch.Tensor):
        a_n = 0.01 * (10 - self.v) / (torch.exp((torch.as_tensor(10.) - self.v) / 10) - 1)
        b_n = 0.125 * torch.exp(-self.v / torch.as_tensor(80))
        a_m = 0.1 * (25 - self.v) / (torch.exp((torch.as_tensor(25) - self.v) / 10) - 1)
        b_m = 4 * torch.exp(-self.v / torch.as_tensor(18))
        b_h = 1 / (torch.exp(torch.as_tensor((30. - self.v) / 10)))
        a_h = 0.07 * torch.exp(-self.v / torch.as_tensor(20))

        g_Na = torch.as_tensor(self.gbar_Na * self.h * self.m ** 3)
        g_K = torch.as_tensor(self.gbar_K * (self.y ** 4))
        I_Na = torch.as_tensor(g_Na * (self.v - self.V_Na))
        I_K = torch.as_tensor(g_K * (self.v - self.V_K))
        I_L = torch.as_tensor(self.gbar_L * (self.v - self.V_L))
        self.v = self.v + (x - I_Na.detach() - I_K.detach() - I_L.detach()) * self.w.sigmoid()
        self.y = self.y + (a_n * (1 - self.y) - b_n * self.y) * self.w.sigmoid()
        self.m = self.m + (a_m * (1 - self.m) - b_m * self.m) * self.w.sigmoid()
        self.h = self.h + (a_h * (1 - self.h) - b_h * self.h) * self.w.sigmoid()