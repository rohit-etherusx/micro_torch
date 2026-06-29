from micro_torch import Tensor


def reset(*nodes):
    for node in nodes:
        node.grad = 0.0


def test_add_backward_propagates_gradients():
    left = Tensor(2)
    right = Tensor(3)
    result = left + right

    result.grad = 1.0
    result._backward()

    assert left.grad == 1.0
    assert right.grad == 1.0


def test_mul_backward_propagates_gradients():
    left = Tensor(2)
    right = Tensor(3)
    result = left * right

    result.grad = 1.0
    result._backward()

    assert left.grad == 3.0
    assert right.grad == 2.0


def test_div_backward_propagates_gradients():
    left = Tensor(8)
    right = Tensor(2)
    result = left / right

    result.grad = 1.0
    result._backward()

    assert left.grad == 0.5
    assert right.grad == -2.0


def test_pow_backward_propagates_gradients():
    base = Tensor(2)
    result = base ** 3

    result.grad = 1.0
    result._backward()

    assert base.grad == 12.0


def test_negation_backward_propagates_gradients():
    value = Tensor(5)
    result = -value

    result.grad = 1.0
    result._backward()

    assert value.grad == -1.0