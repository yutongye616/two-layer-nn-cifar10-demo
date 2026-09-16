# Two-Layer NN — CIFAR-10 Classifier (Demo)

**[Try the live demo →](https://huggingface.co/spaces/Yutong616/two-layer-nn-cifar10)**

A from-scratch two-layer fully-connected neural network (manual forward/backward
pass in PyTorch, no `nn.Module` autograd) trained on CIFAR-10, wrapped in a
Gradio app: upload an image, get a prediction across the 10 CIFAR-10 classes
(plane, car, bird, cat, deer, dog, frog, horse, ship, truck).

## Architecture

```
X ──┐
    ├─> Z1 = XW1 + b1 ──> ReLU ──> A1 ──┐
W1 ─┘                                 ├─> Scores = A1W2 + b2 ──> Softmax ──> L
b1 ───────────────────────────────────┘
W2, b2 ───────────────────────────────^
y ─────────────────────────────────────^
```

Fully connected layer → ReLU → fully connected layer → softmax cross-entropy
loss with L2 regularization, trained via SGD with a hyperparameter grid
search over learning rate, hidden size, regularization strength, weight-init
scale, and learning-rate decay.

This repo holds just the inference/demo app — a self-contained Gradio
interface plus the trained checkpoint. It's a portfolio-facing companion to
a course project; the full training implementation lives in a private repo.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Note on accuracy

This is a demo of the pipeline (image in → prediction out), not a
state-of-the-art model — expect roughly 50% test accuracy across the 10
classes. It works best on simple, centered photos similar to CIFAR-10's
low-resolution training images.
