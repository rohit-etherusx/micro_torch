class Tensor:
    def __init__(self , data):
        self.data = data

    def __add__(self, other):
        return self.data + other.data




if __name__ == "__main__":
    a = Tensor(5)
    b = Tensor(6)
    print(a+b)