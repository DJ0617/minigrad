# MLP Model
from neural_network_library import Module, LinearLayer

class MLP(Module):
    """
    Multi Layer Perceptron
    """

    def __init__(self, dimensions):
        """
        Initialize multiple layers here in the list named self.linear_layers
        """
        assert isinstance(dimensions, list)
        assert len(dimensions) > 2
        self.linear_layers = []
        for i in range(len(dimensions) - 1):
            self.linear_layers.append(LinearLayer(dimensions[i], dimensions[i + 1]))

    def __call__(self, x):
        """
        Args:
            x (2d-list): Two dimensional list of Values with shape [batch_size , nin]

        Returns:
            xout (2d-list): Two dimensional list of Values with shape [batch_size, nout]
        """
        num_l = len(self.linear_layers)
        x_out = [x]
        for n in range(num_l):
            pro_out = self.linear_layers[n](x_out.pop())
            if n == (num_l - 1): return pro_out
            next_input = [[pro_out[i][j].relu() for j in range(len(pro_out[0]))] for i in range(len(pro_out))]
            x_out.append(next_input)
        return x_out

    def parameters(self):
        """
        Get the parameters of each layer

        Args:
            None

        Returns:
            params (list of Values): Parameters of the MLP
        """
        return [p for layer in self.linear_layers for p in layer.parameters()]

    def zero_grad(self):
        """
        Zero out the gradient of each parameter
        """
        for p in self.parameters():
            p.grad = 0