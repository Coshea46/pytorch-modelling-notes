
- **[`torch.nn.Module`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/generated/torch.nn.Module.html%5D\(https://pytorch.org/docs/stable/generated/torch.nn.Module.html\))**

    - **Purpose:** The base class for all neural network modules.

    - **Key Methods:** `forward()`, `train()`, `eval()`, `parameters()`, `to(device)`.

- **[`torch.Tensor.backward()`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/generated/torch.Tensor.backward.html%5D\(https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html\))**

    - **Purpose:** The entry point for the **Autograd** engine. It kicks off the "Inspector" to calculate derivatives.

- **[`torch.optim.Optimizer.step()`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html%5D\(https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html\))**

    - **Purpose:** The "Plumber" logic. It actually performs the weight update based on the gradients.

- **[`torch.optim.Optimizer.zero_grad()`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html%5D\(https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html\))**

    - **Purpose:** Clears the gradient buffers. Essential for preventing "Exploding Gradients" across batches.
