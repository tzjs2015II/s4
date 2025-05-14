import torch

# 继承自自动微分的torch.autograd.Function
class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 仅当输入需要梯度时才保存数据
        if x.requires_grad:
            # 将输入 x 保存到上下文中，供反向传播时使用
            ctx.save_for_backward(x)
        # 阶跃函数：当 x >= 0 时返回 1，否则返回 0，且结果与输入 x 保持相同的数据类型
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        # 计算输入的绝对值
        x_abs = x.abs()
        # 创建一个布尔掩码，标记绝对值大于 1 的位置
        mask = x_abs > 1
        # 当 |x| <= 1 时，导数为 -|x| + 1.0，这是一个关于 |x| 的线性函数，在 x=0 处取得最大值 1，
        # 在 |x|=1 处降为 当 |x| > 1 时，导数为 0（通过 masked_fill_ 方法将对应位置的值设为 0
        grad_x = (grad_output * (-x_abs + 1.0)).masked_fill_(mask, 0)
        return grad_x, None


def piecewise_quadratic_surrogate():
    return piecewise_quadratic.apply
