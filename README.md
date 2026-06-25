# micro_torch

*Tiny tensors, big dreams.*

Welcome to **micro_torch** – the lightweight, no‑frills tensor library that pretends it could be a deep‑learning framework if it tried a little harder. It’s perfect for:

- People who love playing with numbers but hate installing massive dependencies.
- Students who need a quick sanity‑check for arithmetic without pulling in PyTorch.
- Anyone who enjoys a good joke about tensors being *tiny*.

## What does it do?

It defines a single class, `Tensor`, that stores a scalar value and remembers how it was created. Every arithmetic operation (`+`, `-`, `*`, `/`, `**`, unary `-`) returns a **new** `Tensor` that records:

- The resulting `data` (the actual number).
- A list of `parents` – the tensors that participated in the operation.
- The `op` string (e.g. `'+'`, `'-'`, `'*'`, `'/'`, `'pow'`, `'neg'`).

That’s it. No autograd, no GPU support, no fancy broadcasting – just pure Python fun.

## Quick start

```python
from micro_torch import Tensor

# Create a couple of leaf tensors
x = Tensor(5)
y = Tensor(3)

# Do some math – each step gives you a new Tensor
z = (x + y) * (x - y) / (x ** 2)

print(z)          # Tensor(2.6666666666666665)
print(z.parents)  # shows the computation graph
```

## Running the test suite

We ship a fairly exhaustive set of pytest tests. To make sure everything still works after you tinker with the code:

```bash
$ pytest -q
```

All tests should pass – if they don’t, you probably broke the universe (or just a typo).

## Why the name?

Because it’s **micro** – it only handles a single scalar – and **torch** – we like the sound of it. Think of it as a tiny torch you can shine on your calculator.

## Contributing

Feel free to open issues or PRs. Just remember:

1. Keep it tiny. No heavy dependencies.
2. Add a comment explaining why you added something (we love comments).
3. If you add a new feature, please add a test – the test suite is our safety net.

## License

MIT – because we’re generous and you can do whatever you want with it.

---

*If you can’t build this, you probably need a bigger torch. Happy coding!*