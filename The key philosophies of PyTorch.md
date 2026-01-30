
- **Parallel processing** - allows for gpu optimization
- **Abstraction of back-propagation** - means you can get focused more on building a model, rather than having to worry about differentiation.
- **Dynamic Computation**- Very important. Since datasets can be very large we rarely can load everything into the gpu at once. We should only load what we need at a time and destroy what we no longer need.


# The importance of Dynamic Computation

A key feature of PyTorch is the way it handles training by using [[Computational Training Graphs|Dynamic Computational Graphs (Eager Execution)]] to allow for optimal memory usage and hence training times.

