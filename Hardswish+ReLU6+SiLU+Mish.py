import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input):
        return F.silu(input)


def ReLU6(x) -> list:
    x = x.numpy()
    y = []
    for v in x:
        if v <= 0:
            y.append(0)
        elif v > 0 and v <= 6:
            y.append(v)
        else:
            y.append(6)
    return y


class Hardswish(nn.Module):
    def __init__(self):
        super(Hardswish, self).__init__()

    def forward(self, input):
        return F.hardswish(input)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


def Curve(x):
    silu = SiLU()
    y0 = silu(x)

    y1 = ReLU6(x)

    hardswish = Hardswish()
    y2 = hardswish(x)

    mish = Mish()
    y3 = mish(x)

    plt.title('Hardswish+ReLU6+SiLU')
    plt.plot(x, y0, color='green', label='SiLU')
    plt.plot(x, y1, color='blue', label='ReLU6')
    plt.plot(x, y2, '--', color='red', label='Hardswish')
    plt.plot(x, y3, color='purple', label='Mish')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    x = torch.linspace(-10, 10, 10000)
    Curve(x)
