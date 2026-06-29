# micro_torch

*Tiny tensors, big dreams.*

Welcome to **micro_torch** — the lightweight, no-frills tensor library that pretends it could be a real deep-learning framework if it had a better bedtime routine and a little less ego. If you can build this, great. If you can’t, well, that’s usually because you still think gradients are just a personality trait.

## What does it do?

It defines a single class, `Tensor`, that stores a scalar value and remembers how it was created. Every arithmetic operation (`+`, `-`, `*`, `/`, `**`, unary `-`) returns a new `Tensor` that keeps track of its parents, so you can inspect the graph, do a little manual backprop, and pretend you’re one step away from training a model that changes the world.

## What’s new and slightly less embarrassing

A few new features have been added, because apparently the project had to stop living in the past:

- `exp()` for exponential-style calculations
- `tanh()` for that classic activation energy moment
- `backward()` to walk the computation graph and propagate gradients like you actually know what you’re doing
- `zero_grad()` to clear gradients across the graph without needing a full emotional reset
- better gradient handling for core arithmetic and unary operations

So yes, this tiny torch now has a bit more substance. Still small, still silly, but now it can at least fake competence with a straight face.

## Quick start

```python
from micro_torch import Tensor

x = Tensor(5)
y = Tensor(3)

z = (x + y) * (x - y) / (x ** 2)

print(z)          # Tensor(2.6666666666666665)
print(z.parents)  # shows the computation graph
```

## Running the test suite

We ship a fairly exhaustive set of pytest tests, because if you can’t build the thing, you probably don’t know it well enough to break it in a clever way.

```bash
$ pytest -q
```

All tests should pass — unless you’ve been aggressively improvising again.

## Why the name?

Because it’s **micro** — only one scalar at a time — and **torch** — because it likes to act like it belongs in a serious ML lab. Think of it as a tiny torch you can shine on your calculator and call it research.

## Contributing

Feel free to open issues or PRs. Just remember:

1. Keep it tiny. No heavy dependencies.
2. Add a comment explaining why you added something — we love comments.
3. If you add a new feature, add a test. The test suite is our safety net.

## License

MIT — because we’re generous, and apparently committed to tiny neural dreams.

*If you can’t build this, you probably need a bigger torch. Happy coding!*