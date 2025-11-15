import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # forward hook → save activations
        target_layer.register_forward_hook(self.save_activations)

        # backward hook → save gradients
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        class_idx = output.argmax(dim=1).item()

        # backward on predicted class
        output[0, class_idx].backward()

        grads = self.gradients        # (B, C, H, W)
        acts = self.activations       # (B, C, H, W)

        # global average pooling over H, W
        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)  # (B, H, W)

        # detach before converting to numpy
        cam = cam.squeeze().detach().cpu().numpy()

        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx
