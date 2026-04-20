import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Optional
from dataclasses import dataclass
import kornia.augmentation as K
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================

@dataclass
class thermalAugConfig:
    image_size: int = 224

    # Thermal erase
    mask_width_ratio: float = 0.6
    mask_height_ratio: float = 0.2
    max_attempts: int = 5
    erase_prob: float = 0.9

    # Flips
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3

    # Photometric
    brightness_range: tuple = (0.2, 0.7)
    contrast_range: tuple = (0.2, 1)
    thermal_contrast_range: tuple = (0.5, 1.0)

    # Elastic
    elastic_alpha_scale: float = 0.08
    elastic_sigma_scale: float = 0.08

    # Buffer-level probs (ADDED missing ones)
    occlusion_prob: float = 0.4
    brightness_contrast_prob: float = 0.5
    thermal_contrast_prob: float = 0.5
    elastic_transform_prob: float = 0.3

    # Geometric (for Kornia)
    degrees: int = 10
    translate: tuple = (0.1, 0.1)
    scale: tuple = (0.85, 1.15)
    shear: float = 10

    random_affine_prob: float = 0.3
    random_rotation_degrees: int = 5
    random_rotation_degrees_prob: float = 0.5
    random_crop_prob: float = 0.5

    resized_crop_scale: tuple = (0.85, 1.0)
    resized_crop_ratio: tuple = (0.75, 1.25)

    # Normalization
    mean: tuple = (0.24, 0.24, 0.24)
    std: tuple = (0.07, 0.07, 0.07)


# ============================================================
# AUGMENTOR
# ============================================================

class ThermalAugmentor:
    def __init__(self, config: thermalAugConfig = thermalAugConfig()):
        self.cfg = config

    # --------------------------------------------------------
    # Utils
    # --------------------------------------------------------
    def _ensure_batched(self, x):
        if x.dim() == 3:
            return x.unsqueeze(0), False
        return x, True

    def _restore_shape(self, x, is_batched):
        return x if is_batched else x[0]

    # --------------------------------------------------------
    # Horizontal Flip
    # --------------------------------------------------------
    def _horizontal_flip(self, buffer: torch.Tensor, boxes=None):
        x = buffer.clone()

        if torch.rand(1).item() >= self.cfg.horizontal_flip_prob:
            return x

        x = torch.flip(x, dims=[2])

        if boxes is not None:
            width = x.shape[2]
            flipped_boxes = boxes.copy()
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1
        else:
            flipped_boxes = None

        return x

    # --------------------------------------------------------
    # Thermal Erase
    # --------------------------------------------------------
    def _thermal_erase(self, buffer: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.erase_prob:
            return buffer

        x, is_batched = self._ensure_batched(buffer)
        T, h, w, c = x.shape

        mask_w = int(w * self.cfg.mask_width_ratio)
        mask_h = int(h * self.cfg.mask_height_ratio)

        cx1, cy1 = int(w * 0.3), int(h * 0.3)
        cx2, cy2 = int(w * 0.7), int(h * 0.7)

        for _ in range(self.cfg.max_attempts):
            x1 = torch.randint(0, max(1, w - mask_w), (1,)).item()
            y1 = torch.randint(0, max(1, h - mask_h), (1,)).item()

            x2, y2 = x1 + mask_w, y1 + mask_h

            overlaps = (
                x2 > cx1 and x1 < cx2 and
                y2 > cy1 and y1 < cy2
            )

            if not overlaps:
                x[:, y1:y2, x1:x2, :] = 0
                return self._restore_shape(x, is_batched)

        x[:, y1:y1+mask_h, x1:x1+mask_w, :] = 0
        return self._restore_shape(x, is_batched)

    # --------------------------------------------------------
    # Brightness + Contrast
    # --------------------------------------------------------
    def _brightness_contrast(self, buffer: torch.Tensor) -> torch.Tensor :
        x, is_batched = self._ensure_batched(buffer)
        x = x.float()

        b_min, b_max = self.cfg.brightness_range
        c_min, c_max = self.cfg.contrast_range

        brightness = torch.empty(1).uniform_(b_min, b_max).item()
        contrast = torch.empty(1).uniform_(c_min, c_max).item()

        for t in range(x.shape[0]):
            mean = x[t].mean()
            x[t] = brightness * x[t] + (contrast - 1.0) * (x[t] - mean) + mean

        return self._restore_shape(torch.clamp(x, 0, 255).to(torch.uint8), is_batched)

    # --------------------------------------------------------
    # Thermal Contrast
    # --------------------------------------------------------
    def _thermal_contrast(self, buffer: torch.Tensor) -> torch.Tensor:
        x, is_batched = self._ensure_batched(buffer)
        x = x.float()

        c_min, c_max = self.cfg.thermal_contrast_range
        factor = torch.empty(1).uniform_(c_min, c_max).item()

        x = torch.clamp(x * factor, 0, 255).to(torch.uint8)
        return self._restore_shape(x, is_batched)

    # --------------------------------------------------------
    # Elastic Transform
    # --------------------------------------------------------
    def _elastic_transform(self, buffer: torch.Tensor) -> torch.Tensor:
        arr = buffer.clone().cpu().numpy()
        is_batched = (arr.ndim == 4)

        if not is_batched:
            arr = np.expand_dims(arr, 0)

        T, h, w, c = arr.shape

        alpha = self.cfg.image_size * self.cfg.elastic_alpha_scale
        sigma = self.cfg.image_size * self.cfg.elastic_sigma_scale

        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.vstack([(y_coords + dy).ravel(), (x_coords + dx).ravel()])

        out = np.zeros_like(arr)

        for t in range(T):
            for ch in range(c):
                out[t, :, :, ch] = map_coordinates(
                    arr[t, :, :, ch], indices, order=1, mode="reflect"
                ).reshape(h, w)

        out = torch.from_numpy(np.clip(out, 0, 255).astype(np.uint8))
        return out if is_batched else out[0]

    # ========================================================
    # BUFFER PIPELINE (NEW - INTEGRATED)
    # ========================================================
    def _augment_buffer(self, buffer: torch.Tensor) -> torch.Tensor:
            """
            Apply multiple stochastic augmentations independently
            """
            buffers = self.resize_buffer(buffer, size=(224, 224))

            if torch.rand(1).item() < self.cfg.occlusion_prob:
                buffers = self._thermal_erase(buffers)

            if torch.rand(1).item() < self.cfg.brightness_contrast_prob:
                buffers = self._brightness_contrast(buffers)
              
            if torch.rand(1).item() < self.cfg.thermal_contrast_prob:
                buffers = self._thermal_contrast(buffers)

            if torch.rand(1).item() < self.cfg.elastic_transform_prob:
                buffers = self._elastic_transform(buffers)
                buffers = self._elastic_transform(buffers)
                
            if torch.rand(1).item() < self.cfg.horizontal_flip_prob:
                buffers = self._horizontal_flip(buffers)
          
            return buffers
                buffers = self._horizontal_flip(buffers)
          
            return buffers


    # ========================================================
    # GEOMETRIC TRANSFORMS (KORNIA)
    # ========================================================
    def _apply_geometric_transforms(self, buffer: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        T,_,_,_ = buffer.shape
        buffer = self.resize_buffer(buffer)
        buffer_tv = buffer.permute(0, 3, 1, 2)

        aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=self.cfg.degrees,
                translate=self.cfg.translate,
                scale=self.cfg.scale,
                shear=self.cfg.shear,
                p=self.cfg.random_affine_prob
            ),
            K.RandomRotation(
                degrees=self.cfg.random_rotation_degrees,
                p=self.cfg.random_rotation_degrees_prob
            ),
            K.RandomResizedCrop(
                size=(self.cfg.image_size, self.cfg.image_size),
                scale=self.cfg.resized_crop_scale,
                ratio=self.cfg.resized_crop_ratio,
                p=self.cfg.random_crop_prob
            ),
            K.RandomAutoContrast(p=0.3),
            K.RandomHorizontalFlip(p=self.cfg.horizontal_flip_prob),
            K.RandomVerticalFlip(p=self.cfg.vertical_flip_prob),
            same_on_batch=False,
            data_keys=["input"],
        )

        result = aug(buffer_tv)

        no_crop_mask = (torch.rand(T) >= self.cfg.random_crop_prob)
        if no_crop_mask.any():
            result[no_crop_mask] = torch.stack([
                TF.resize(img, (self.cfg.image_size, self.cfg.image_size))
                for img in result[no_crop_mask]
            ])

        return result.permute(0, 2, 3, 1)
     # --------------------------------------------------------
    
    # PUBLIC ENTRY POINT
    # --------------------------------------------------------
    def __call__(self, buffer: torch.Tensor,is_shared: bool = True) -> torch.Tensor:
        """
        Main entry: user calls pipeline(buffer)
        """
        buffer = buffer.clone()  # safety (avoid inplace bugs)
        # 1. global augmentations (optional)
        if is_shared:
            buffer = self._apply_geometric_transforms(buffer)
        else:
            # 2. geometric transforms (optional)
            buffer = self._augment_buffer(buffer)

        output = self._tensor_normalize_inplace(buffer, self.cfg.mean, self.cfg.std)     

        return output
    
    def _tensor_normalize_inplace(self,tensor, mean, std):
            if tensor.dtype == torch.uint8:
              tensor = tensor.float()

            tensor = tensor.permute(3, 0, 1, 2)  # [C, T, H, W]

            C, T, H, W = tensor.shape

            # ensure mean/std are tensors on correct device + dtype
            mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

            # scale if needed (more stable check)
            if tensor.max() > 1.5:
                mean = mean * 255.0
                std = std * 255.0

            tensor = tensor.view(C, -1).permute(1, 0)

            # avoid division by zero (VERY IMPORTANT)
            std = torch.where(std == 0, torch.ones_like(std), std)

            tensor = tensor.sub(mean).div(std)

            tensor = tensor.permute(1, 0).view(C, T, H, W)

            return tensor
        
    def resize_buffer(self, buffer:torch.Tensor, size=(224, 224)):
        # buffer: [T, H, W, 3]

        # convert to [T, C, H, W]
        buffer = buffer.permute(0, 3, 1, 2)

        # resize each frame (or whole batch at once)
        buffer = F.interpolate(buffer, size=size, mode="bilinear", align_corners=False)

        # back to [T, H, W, 3]
        buffer = buffer.permute(0, 2, 3, 1)

        return buffer
