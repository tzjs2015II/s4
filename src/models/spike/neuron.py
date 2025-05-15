from torch import nn
import torch
import os

"""
- **SDNNeuron功能**：实现论文中的 **代理动态网络（SDN）**，通过预训练的轻量级卷积网络（`self.model`）直接预测膜电位，避免传统LIF神经元的迭代计算。
- **膜电位预测**：`pred` 方法使用预训练模型并行预测所有时间步的膜电位，时间复杂度为 $ O(1) $ （见论文中“并行计算原理”）。
- **脉冲生成**：将预测的膜电位与输入电流叠加（`mem + x`），减去阈值（此处固定为1.0）后，通过代理梯度函数（如分段二次函数）生成脉冲，模拟LIF神经元的“膜电位累积-阈值触发”过程。
"""
class SDNNeuron(nn.Module):
    ckt_path = "mem_pred_d8k8_c3(relu(c2(c1(x))+c1(x)))_125.pkl"

    def __init__(self, surrogate_function):
        super().__init__()
        self.surrogate_function = surrogate_function
        ckt_path = os.path.join(os.path.dirname(__file__), self.ckt_path)
        self.model = torch.jit.load(ckt_path).eval()

    def forward(self, x):
        # 预测膜电位（前向传播）
        mem = self.pred(x)  # 通过SDN模型预测膜电位
         # 计算脉冲：代理梯度函数处理（膜电位 + 输入 - 阈值，此处阈值设为1.0）
        s = self.surrogate_function(mem + x - 1.0)
        return s

    @torch.no_grad() # 无梯度模式（推理阶段预测膜电位）
    def pred(self, x):
        shape = x.shape
        L = x.size(-1)
        # 将输入展平为(batch*channels, 1, L)以适配卷积模型,后恢复初始形状
        return self.model(x.detach().view(-1, 1, L)).view(shape)

"""
- **迭代计算膜电位**：逐时间步更新膜电位，符合LIF神经元的“漏积分-点火-重置”动态（见论文中“LIF神经元模型”部分）。
- **反向传播兼容性**：使用代理梯度函数（`surrogate_function`）计算脉冲梯度，解决阶跃函数不可微问题（见论文中“代理梯度训练”部分）。
- **时间复杂度**：逐时间步循环，复杂度为 $ O(L) $ ，长序列（如L=8K）时效率较低（论文中对比SDN的加速效果）。
"""
class BPTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0]) # 初始化膜电位u为全零（形状与输入特征维度匹配）
        out = [] # 存储各时间步的脉冲序列
        for i in range(x.size(-1)):# 逐时间步迭代（序列长度L）
            # 膜电位动态：u = τ * u_prev + (1-τ) * 输入电流（此处(1-τ)=0.875）
            u = u * self.tau + x[..., i]
            # 脉冲生成：代理梯度函数处理（u - 阈值）
            s = self.surrogate_function(u - self.vth)
            out.append(s)
            # 硬重置：若发放脉冲（s=1），u重置为v_r（否则保留当前u）
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)# 堆叠所有时间步的脉冲，形状为(B, H, L)

"""
- **梯度截断**：`u = u.detach() * self.tau` 强制断开前一步膜电位的梯度，仅计算当前时间步的局部梯度，减少反向传播时的梯度消失问题（适用于长序列，但牺牲了时间依赖建模能力）。
- **与BPTT的对比**：BPTT保留完整梯度链（复杂度 $ O(L) $ ），SLTTN通过截断梯度降低计算量，但可能影响长程依赖建模（论文中作为传统方法对比SDN的加速效果）。
"""
class SLTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            # 关键区别：使用detach()断开历史梯度，仅计算当前步局部梯度
            u = u.detach() * self.tau + x[..., i]
            s = self.surrogate_function(u - self.vth)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)


registry = {
    "sdn": SDNNeuron,
    "bptt": BPTTNueron,
    "sltt": SLTTNueron,
}
