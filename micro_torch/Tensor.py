class Tensor:
    def __init__(self, data, parents=None, op=None):
        self.data = data
        self.parents = parents or []
        self.op = op

    def _ensure_tensor(self, value):
        if isinstance(value, Tensor):
            return value
        
        return Tensor(value)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        # return self.data + other.data
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data + other.data,
            parents=[self, other],
            op='+')
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data - other.data,
            parents=[self, other],
            op='-')
    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data * other.data,
            parents=[self, other],
            op='*')
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return Tensor(
            data=self.data / other.data,
            parents=[self, other],
            op='/'
        )
    def __rtruediv__(self, other):
        return self / other


if __name__ == "__main__":
    a = Tensor(5)

    print(a + 3)
    print(3 + a)

    print(a + 2.5)
    print(2.5 + a)

    print(a + Tensor(10))