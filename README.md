# minigrad
a mini autogradient engine using backward propagation technique plus a simple neural network model library

In MiniGrad, an Auto_Gradient class named `Scalar`. The basic idea is to store the existing computational map during the creation of each Value, and calculate the gradient using backpropagation when one of the Value calls `backward()` method. The `backward()` function will arange the computational graph and backpropagate the gradients.

The design of the mini-grad structure are based on the work https://github.com/karpathy/micrograd 
