"""
MediScan AI - Production Inference Engine
Handles model loading, image preprocessing, prediction, and Grad-CAM generation.
"""

from __future__ import annotations

import io
import base64
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Architecture (must match training definition exactly)
# ---------------------------------------------------------------------------

class MediScanModel(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        backbone         = efficientnet_b4(weights=None)
        self.features    = backbone.features
        self.avgpool     = backbone.avgpool
        in_features      = backbone.classifier[1].in_features
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Generates Gradient-weighted Class Activation Maps.
    Hooks into the final convolutional block of EfficientNetB4.
    """
    def __init__(self, model: MediScanModel):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer     = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        return cam.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Singleton-style inference engine.
    Loads the model once and serves all prediction requests.
    """

    CLASSES = ["NORMAL", "PNEUMONIA"]
    IMAGE_SIZE = 380

    RISK_MAP = {
        "NORMAL": {
            "high"     : ("LOW",      "No radiographic signs of pneumonia detected."),
            "moderate" : ("MODERATE", "Findings suggest normal presentation. Clinical correlation advised."),
            "low"      : ("MODERATE", "Result is inconclusive. Radiologist review recommended."),
        },
        "PNEUMONIA": {
            "high"     : ("HIGH",     "Radiographic signs consistent with pneumonia. Immediate clinical evaluation required."),
            "moderate" : ("HIGH",     "Findings suspicious for pneumonia. Clinical and laboratory correlation required."),
            "low"      : ("MODERATE", "Possible early pneumonia or other infiltrate. Further workup recommended."),
        },
    }

    def __init__(self):
        self._model:   Optional[MediScanModel] = None
        self._gradcam: Optional[GradCAM]       = None
        self._device:  torch.device            = torch.device("cpu")

        self._transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def load(self, model_path: str) -> None:
        """Load model weights from checkpoint file."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint   = torch.load(model_path, map_location=self._device, weights_only=False)

        self._model = MediScanModel(num_classes=2, dropout=0.4)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._model.to(self._device)

        self._gradcam = GradCAM(self._model)

        val_auc = checkpoint.get("val_auc", "N/A")
        val_acc = checkpoint.get("val_acc", "N/A")
        logger.info(
            "Model loaded — device=%s  val_auc=%.4f  val_acc=%.2f%%",
            self._device, val_auc, val_acc
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run full inference pipeline on raw image bytes.

        Returns a structured result dict containing:
          - predicted_class   : str
          - confidence        : float (0-100)
          - all_probabilities : dict[str, float]
          - risk_level        : 'LOW' | 'MODERATE' | 'HIGH'
          - clinical_note     : str
          - gradcam_overlay   : base64-encoded PNG
          - model_version     : str
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Decode and preprocess image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self._transform(pil_image).unsqueeze(0).to(self._device)

        # Forward pass
        with torch.no_grad():
            logits = self._model(input_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx   = int(np.argmax(probs))
        predicted_class = self.CLASSES[predicted_idx]
        confidence      = float(probs[predicted_idx]) * 100.0

        # Grad-CAM (requires gradients)
        gradcam_b64 = self._generate_gradcam_overlay(input_tensor, pil_image, predicted_idx)

        # Risk assessment
        conf_band = "high" if confidence >= 80 else ("moderate" if confidence >= 60 else "low")
        risk_level, clinical_note = self.RISK_MAP[predicted_class][conf_band]

        return {
            "predicted_class"   : predicted_class,
            "confidence"        : round(confidence, 2),
            "all_probabilities" : {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(self.CLASSES, probs)
            },
            "risk_level"        : risk_level,
            "clinical_note"     : clinical_note,
            "gradcam_overlay"   : gradcam_b64,
            "model_version"     : "MediScan-EfficientNetB4-v5",
        }

    def _generate_gradcam_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        class_idx: int
    ) -> str:
        """Generate Grad-CAM heatmap overlaid on original image, return as base64 PNG."""
        try:
            # Re-run forward pass with gradient tracking
            tensor_grad = input_tensor.clone().requires_grad_(True)
            cam = self._gradcam.generate(tensor_grad, class_idx)

            # Suppress border artifacts: zero out outer 10% margin where
            # normalization padding creates false high-gradient corners
            h, w = cam.shape
            bh, bw = max(1, int(h * 0.10)), max(1, int(w * 0.10))
            border_mask = np.ones_like(cam)
            border_mask[:bh, :]  = 0
            border_mask[-bh:, :] = 0
            border_mask[:, :bw]  = 0
            border_mask[:, -bw:] = 0
            cam = cam * border_mask

            # Re-normalize after masking so the lung region fills 0..1
            if cam.max() > 0:
                cam = cam / cam.max()

            # Gaussian smooth to reduce noise and make regions more contiguous
            from scipy.ndimage import gaussian_filter
            cam = gaussian_filter(cam, sigma=2)
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize cam to original image size
            img_w, img_h = original_image.size
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize((img_w, img_h), Image.BILINEAR),
                dtype=np.float32
            ) / 255.0

            # Apply jet colormap (matches matplotlib plt.cm.jet used in training notebook)
            # jet: 0.0=blue, 0.25=cyan, 0.5=green, 0.75=yellow, 1.0=red
            c = cam_resized
            r = np.clip(1.5 - np.abs(c * 4.0 - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(c * 4.0 - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(c * 4.0 - 1.0), 0.0, 1.0)

            # Convert original image to RGB numpy array
            orig_np = np.array(original_image.convert("RGB"), dtype=np.float32) / 255.0

            # Blend: 40% original + 60% jet heatmap (matches notebook visual style)
            alpha = 0.6
            blended = np.zeros((img_h, img_w, 3), dtype=np.float32)
            blended[:,:,0] = (1 - alpha) * orig_np[:,:,0] + alpha * r
            blended[:,:,1] = (1 - alpha) * orig_np[:,:,1] + alpha * g
            blended[:,:,2] = (1 - alpha) * orig_np[:,:,2] + alpha * b
            blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

            # Encode to base64
            buf = io.BytesIO()
            Image.fromarray(blended).save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as exc:
            logger.warning("Grad-CAM generation failed: %s", exc)
            # Return original image as fallback
            buf = io.BytesIO()
            original_image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")


# Module-level singleton
engine = InferenceEngine()