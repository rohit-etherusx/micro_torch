


class Tensor:
    def __init__(self, data, parents=None, op=None):
        
        self.data = data                        # scalar payload (int or float)
        self.parents = parents or []            # list of parent Tensors (empty for leaf tensors)
        self.op = op                            # operation name that produced this tensor (None for leaves)
        self._backward = lambda : None          # the back_propagation functions calls
        self.grad = 0.0                         # the node gradient 
 
  # wrap scalars into a Tensor; return Tensor unchanged
    def _ensure_tensor(self, value):
        if isinstance(value, Tensor):
            return value
        return Tensor(value)

    # concise string representation
    def __repr__(self):
        return f"Tensor({self.data})"

    # primitive add operation
    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            data=self.data + other.data,
            parents=[self, other],
            op='+')

        def _backward() : 
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward

        return out

    def __radd__(self, other):
        # support scalar + Tensor
        return self + other

    # primitive subtraction (kept as original implementation)
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = self + (-other)
        out.op = '-'
        return out
        # explicit subtraction (commented to preserve original form):
        # return Tensor(data=self.data - other.data, parents=[self, other], op='-')

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other - self

    # multiplication
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            data=self.data * other.data,
            parents=[self, other],
            op='*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # true division
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        out =  Tensor(
            data=self.data / other.data,
            parents=[self, other],
            op='/'
        )
        def _backward():
            self.grad += (1/other.data)*out.grad
            other.grad +=  (-self.data/(other.data)**2)*out.grad
        
        out._backward = _backward

        return out

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other / self

    # power
    def __pow__(self, other):
        assert isinstance(other, (int, float)),  "MicroTorch currently supports only scalar exponents"
        out =  Tensor(
            data=self.data ** other,
            parents=[self],
            op='pow'
        )
        def _backward():
            self.grad += out.grad*(other*(self.data**(other-1)))
        out._backward = _backward
        return out

    # negation
    def __neg__(self):
        out =  Tensor(
            data=self.data * -1,
            parents=[self],
            op='neg'
        )

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward
        return out

if __name__ == "__main__":
    a = Tensor(10)
    b = -a

    print(a)
    print(b)
