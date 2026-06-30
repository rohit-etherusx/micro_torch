import numpy as np 

# print(np.__version__)

class Tensor:
    def __init__(self ,data ,parents = None ,op = None):
        self.data = np.array(data)
        self.parents = parents or []
        self.op = op
        self.grad = np.zeros_like(self.data)
        self._backward = lambda:None
        

    def _ensure_tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)
    
        
    def __repr__(self):
        return f"Tensor({self.data})"
    
    
    
    def __add__(self,other):
        other = self._ensure_tensor(other=other)
        
        out = Tensor(
            data = self.data + other.data,
            parents = [self,other],
            op = '+')
        
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
            
        out._backward = _backward
        return out 
    
    def __radd__(self, other):
        return self+other
    

    
    def __sub__(self,other):
        other = self._ensure_tensor(other)
        
        out = Tensor(
            data  = self.data - other.data,
            parents = [self,other],
            op  = '-')
        
        def _backward():
            self.grad += out.grad 
            other.grad += -out.grad 
            
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return self._ensure_tensor(other) - self
    
    
    
    def __mul__(self,other):
        other = self._ensure_tensor(other)
        
        out = Tensor(
            data = self.data * other.data,
            parents = [self,other],
            op = '*' )
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
            
        out._backward = _backward            
        return out
    
    def __rmul__(self, other):
        other = self._ensure_tensor(other)
        return other * self
    
    
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        
        out = Tensor(
            data = self.data / other.data,
            parents = [self,other],
            op = "/"
        )
        
        def _backward():
            self.grad += (1/other.data)*out.grad
            other.grad += (-self.data/(other.data**2))*out.grad
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        return self._ensure_tensor(other)/self
    
    
    def __pow__(self, other):
        pass
    
    def __rpow__(self, other):
        pass

if __name__ == "__main__":

    print("===== ADD =====")

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])

    c = a + b

    c.grad = np.ones_like(c.data)
    c._backward()

    print("c =", c.data)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)

    print()

    print("===== SUB =====")

    a = Tensor([10,20,30])
    b = Tensor([1,2,3])

    c = a - b

    c.grad = np.ones_like(c.data)
    c._backward()

    print("c =", c.data)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)

    print()

    print("===== MUL =====")

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])

    c = a * b

    c.grad = np.ones_like(c.data)
    c._backward()

    print("c =", c.data)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)

    print()

    print("===== DIV =====")

    a = Tensor([10,20,30])
    b = Tensor([2,4,5])

    c = a / b

    c.grad = np.ones_like(c.data)
    c._backward()

    print("c =", c.data)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)