class Tensor:
    def __init__(self, data, parents=None, op=None):
        # scalar payload (int or float)
        self.data = data
        # list of parent Tensors (empty for leaf tensors)
        self.parents = parents or []
        # operation name that produced this tensor (None for leaves)
        self.op = op

    def _ensure_tensor(self, value):
        # wrap scalars into a Tensor; return Tensor unchanged
        if isinstance(value, Tensor):
            return value

        return Tensor(value)

    # concise string representation
    def __repr__(self):
        return f"Tensor({self.data})"

    # primitive add operation
    def __add__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data + other.data,
            parents=[self, other],
            op='+')

    def __radd__(self, other):
        # support scalar + Tensor
        return self + other

    # primitive subtraction (kept as original implementation)
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return self + (-other)
        # explicit subtraction (commented to preserve original form):
        # return Tensor(data=self.data - other.data, parents=[self, other], op='-')

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other - self

    # multiplication
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data * other.data,
            parents=[self, other],
            op='*')

    def __rmul__(self, other):
        return self * other

    # true division
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data / other.data,
            parents=[self, other],
            op='/'
        )

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other / self

    # power
    def __pow__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data ** other.data,
            parents=[self, other],
            op='pow'
        )

    # negation
    def __neg__(self):
        return Tensor(
            data=self.data * -1,
            parents=[self],
            op='neg'
        )


if __name__ == "__main__":
    a = Tensor(10)
    b = -a

    print(a)
    print(b)
