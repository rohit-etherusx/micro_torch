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
        other = self._ensure_tensor(other)
        return other - self

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
        other = self._ensure_tensor(other)
        return  other / self

    def __pow__(self,other):
        other = self._ensure_tensor(other)
        return Tensor(
            data = self.data**other.data,
            parents = [self,other],
            op = 'pow'
        )

    def __neg__(self):
        return Tensor(
            data = self.data*-1,
            parents = [self],
            op = 'neg'
        )
    

if __name__ == "__main__":
    a = Tensor(10)
    b = -a
    
    print(a)
    print(b)