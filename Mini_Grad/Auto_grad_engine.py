# Auto Grad Engine using backward propagation

class Scalar:
    """
    Basic unit of storing a single scalar value and its gradient
    """

    def __init__(self, data, _children=()):
        """

        """
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        """
        Args:
            other (Any): Node to add with the class

        Returns:
            out (callable): Function to referesh the gradient
        """
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiplication operation
        """
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Power operation
        """
        assert isinstance(other, (int, float))
        out = Scalar(self.data ** other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        ReLU activation function applied to the current Value
        """
        out = Scalar(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """
        Exponentiate the current Value (e.g. e ^ Value(0) = Value(1))
        """
        out = Scalar(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        """
        Take the natural logarithm (base e) of the current Value
        """
        out = Scalar(np.log(self.data), (self,))

        def _backward():
            self.grad += self.data ** (-1) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Run backpropagation from the current Value
        """

        def topo_list_builder():

            topo, node_visit = list(), set()

            def add_node(node):
                if node not in node_visit:
                    node_visit.add(node)
                    for children in node._prev:
                        add_node(children)
                    topo.append(node)

            add_node(self)

            return topo

        self.grad = 1
        topo = topo_list_builder()
        topo.reverse()
        for v in topo:
            v._backward()

    def __neg__(self):
        """
        Negate the current Value
        """
        return self * -1

    def __radd__(self, other):
        """
        Reverse addition operation
        """
        return self + other

    def __sub__(self, other):
        """
        Subtraction operation
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Reverse subtraction operation
        """
        return other + (-self)

    def __rmul__(self, other):
        """
        Reverse multiplication operation
        """
        return self * other

    def __truediv__(self, other):
        """
        Division operation
        """
        return self * other ** -1

    def __rtruediv__(self, other):
        """
        Reverse diction operation
        """
        return other * self ** -1

    def __repr__(self):
        """
        Class representation
        """
        return f"Value(data={self.data}, grad={self.grad})"