# Linear Layer Neural Network
import random
from Auto_grad_engine import Scalar


class Module:
    """
    Base Model Module
    """

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class LinearLayer(Module):
    """
    Linear Layer
    """

    def __init__(self, nin, nout):
        self.w = []
        for i in range(nin):
            w_tmp = [Scalar(random.uniform(-1, 1)) for j in range(nout)]
            self.w.append(w_tmp)
        self.b = [Scalar(0) for i in range(nout)]
        self.nin = nin
        self.nout = nout

    def __call__(self, x):
        """
        Args:
            x (2d-list): Two dimensional list of Values with shape [batch_size , nin]

        Returns:
            xout (2d-list): Two dimensional list of Values with shape [batch_size, nout]
        """

        batch_size = len(x)

        w_dot_x = [[sum(xi * wi for xi, wi in zip(xr, wr)) for wr in zip(*self.w)] for xr in x]

        out = [[w_dot_x[i][j] + self.b[j] for j in range(self.nout)] for i in range(batch_size)]

        return out

    def parameters(self):
        """
        Get the list of parameters in the Linear Layer

        Args:
            None

        Returns:
            params (list): List of parameters in the layer
        """
        return [p for row in self.w for p in row] + [p for p in self.b]