"""
Gradio demo for Hugging Face Spaces.

Self-contained: reimplements the tiny forward pass directly instead of
importing two_layer_net.py, so the Space doesn't need torchvision/matplotlib
(those are only used for CIFAR-10 downloading/plotting during training).
"""
import numpy as np
import torch
import gradio as gr

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

params = torch.load("demo_checkpoint.pt", map_location="cpu")
mean_image = torch.load("demo_mean_image.pt", map_location="cpu")  # shape (1, 3, 1, 1)


def forward(x: torch.Tensor) -> torch.Tensor:
    hidden = (x @ params["W1"] + params["b1"]).clamp(min=0)
    return hidden @ params["W2"] + params["b2"]


def predict(image):
    if image is None:
        return None
    img = image.convert("RGB").resize((32, 32))
    x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    x = (x.unsqueeze(0) - mean_image).reshape(1, -1)
    probs = torch.softmax(forward(x), dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="Two-Layer NN — CIFAR-10 Classifier",
    description=(
        "A from-scratch two-layer fully-connected network (manual forward/backward "
        "pass, no autograd) trained on CIFAR-10. This is a demo of the pipeline, "
        "not a high-accuracy model — around 51% test accuracy across 10 classes: "
        + ", ".join(CLASSES) + ". Works best on simple, centered photos of those "
        "categories, similar to the low-resolution CIFAR-10 training images."
    ),
)

if __name__ == "__main__":
    demo.launch()
