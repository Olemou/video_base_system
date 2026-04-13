import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Optional, Tuple


class ThermalAugmentor:
    def __init__(
        self,
        image_size: int = 224,
    ):
        self.image_size = image_size

    def horizontal_flip(self, images: torch.Tensor, prob: float = 0.5, boxes=None) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Perform horizontal flip on the given images and corresponding boxes.
        
        Args:
            images: Input tensor in format [T, H, W, C] (PyTorch tensor)
            prob (float): probability to flip the images.
            boxes (ndarray or None): optional. Corresponding boxes to images.
                Dimension is `num boxes` x 4.
        
        Returns:
            images: flipped images (same format as input)
            flipped_boxes: the flipped boxes (or None if no boxes provided)
        """
        # Make a copy to avoid modifying original
        x = images.clone()
        
        # Remember if input was batched (has time dimension)
        is_batched = (x.dim() == 4)
        
        if boxes is None:
            flipped_boxes = None
        else:
            flipped_boxes = boxes.copy()

        if torch.rand(1).item() < prob:
            print("Flipping images")
            
            # Flip horizontally - for [T, H, W, C] format, flip along width dimension (index 2)
            x = torch.flip(x, dims=[2])  # Flip along W dimension
            
            # Get width for box transformation
            width = x.shape[2]  # Width is at index 2 in [T, H, W, C]
            
            if boxes is not None:
                # Flip box coordinates: [x1, y1, x2, y2] -> [width-x2-1, y1, width-x1-1, y2]
                flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1
        
        return x, flipped_boxes

    def thermal_erase(
        self,
        img: torch.Tensor,
        mask_width_ratio: float = 0.6,
        mask_height_ratio: float = 0.2,
        max_attempts: int = 5,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Apply black rectangle occlusion avoiding center region.
        
        Args:
            img: Input tensor in format [T, H, W, C] (PyTorch tensor)
            mask_width_ratio: Width of mask relative to image width
            mask_height_ratio: Height of mask relative to image height
            max_attempts: Maximum attempts to place mask avoiding center
            debug: Print debug information
        
        Returns:
            Augmented tensor in same format [T, H, W, C]
        """
        # Handle input that could be single image or batch [T, H, W, C]
        x = img.clone()  # Make a copy to avoid modifying original
        
        # Check if input is batched [T, H, W, C]
        if x.dim() == 4:
            T, h, w, c = x.shape
            if c not in [1, 3]:
                raise ValueError(f"Unexpected channel dimension: {c}")
            is_batched = True
        elif x.dim() == 3:
            # Single image [H, W, C]
            h, w, c = x.shape
            x = x.unsqueeze(0)  # Add batch dimension
            is_batched = False
            T = 1
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")

        if debug:
            print(f"Input shape: {x.shape}, h={h}, w={w}, c={c}")

        # --- Validate hyperparameters ---
        if not (0 < mask_width_ratio <= 1):
            raise ValueError(f"mask_width_ratio must be in (0,1], got {mask_width_ratio}")
        if not (0 < mask_height_ratio <= 1):
            raise ValueError(f"mask_height_ratio must be in (0,1], got {mask_height_ratio}")

        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >=1, got {max_attempts}")

        mask_w = int(w * mask_width_ratio)
        mask_h = int(h * mask_height_ratio)

        if debug:
            print(f"Mask size: {mask_w}x{mask_h}")

        # Define center region (we want to AVOID placing mask here)
        center_x1, center_y1 = int(w * 0.3), int(h * 0.3)
        center_x2, center_y2 = int(w * 0.7), int(h * 0.7)

        if debug:
            print(f"Center region: x=[{center_x1}, {center_x2}], y=[{center_y1}, {center_y2}]")

        mask_applied = False
        for attempt in range(max_attempts):
            # Random position for top-left corner of mask
            x1 = torch.randint(0, max(1, w - mask_w), (1,)).item()
            y1 = torch.randint(0, max(1, h - mask_h), (1,)).item()
            x2 = x1 + mask_w
            y2 = y1 + mask_h

            # Check if mask overlaps with center region
            overlaps_center = (
                x2 > center_x1 and x1 < center_x2 and y2 > center_y1 and y1 < center_y2
            )

            if debug:
                print(f"Attempt {attempt+1}: mask at ({x1},{y1}) to ({x2},{y2}), overlaps_center={overlaps_center}")

            # We WANT to avoid the center, so only apply if it DOES NOT overlap
            if not overlaps_center:
                # Apply black mask to all frames
                if c == 1:
                    x[:, y1:y2, x1:x2, 0] = 0
                else:
                    x[:, y1:y2, x1:x2, :] = 0
                mask_applied = True
                if debug:
                    print(f"✓ Mask applied at position ({x1},{y1})")
                break

        # If we couldn't find a non-overlapping position, apply it anyway
        if not mask_applied:
            if debug:
                print("⚠ Could not find non-overlapping position, applying random mask anyway")
            x1 = torch.randint(0, max(1, w - mask_w), (1,)).item()
            y1 = torch.randint(0, max(1, h - mask_h), (1,)).item()
            if c == 1:
                x[:, y1:y1+mask_h, x1:x1+mask_w, 0] = 0
            else:
                x[:, y1:y1+mask_h, x1:x1+mask_w, :] = 0

        # Clip values to valid range
        x = torch.clamp(x, 0, 255).to(torch.uint8)

        # Return in the same format as input
        if not is_batched:
            x = x[0]  # Remove batch dimension if input wasn't batched
        
        return x

    def brightness_contrast(
        self,
        img: torch.Tensor,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Adjust brightness and contrast of thermal images.
        
        Args:
            img: Input tensor in format [T, H, W, C] (PyTorch tensor)
            brightness: Brightness factor (if None, random from 0.8-1.4)
            contrast: Contrast factor (if None, random from 0.2-1.2)
        
        Returns:
            Augmented tensor in same format
        """
        # Handle input that could be single image or batch [T, H, W, C]
        arr = img.clone().float()
        is_batched = (arr.dim() == 4)
        
        if not is_batched:
            arr = arr.unsqueeze(0)  # Add batch dimension

        # Validate or randomize parameters
        if brightness is None:
            brightness = torch.empty(1).uniform_(0.8, 1.4).item()
        if contrast is None:
            contrast = torch.empty(1).uniform_(0.2, 1.2).item()

        if brightness <= 0 or contrast <= 0:
            raise ValueError(f"Brightness and contrast must be positive, got brightness={brightness}, contrast={contrast}")

        # Apply brightness and contrast adjustment per frame
        for t in range(arr.shape[0]):
            mean = arr[t].mean()
            arr[t] = brightness * arr[t] + (contrast - 1.0) * (arr[t] - mean) + mean

        arr = torch.clamp(arr, 0, 255).to(torch.uint8)

        if not is_batched:
            arr = arr[0]

        return arr

    def thermal_contrast(self, img: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Simple contrast adjustment for thermal images.
        
        Args:
            img: Input tensor in format [T, H, W, C] (PyTorch tensor)
            alpha: Contrast factor (if None, random from 0.5-1.0)
        
        Returns:
            Augmented tensor in same format
        """
        if alpha is not None and alpha > 1:
            raise ValueError(f"alpha must be <= 1 to increase contrast, got {alpha}")

        # Handle input that could be single image or batch [T, H, W, C]
        x_arr = img.clone().float()
        is_batched = (x_arr.dim() == 4)

        if not is_batched:
            x_arr = x_arr.unsqueeze(0)

        # Determine the contrast scaling factor
        if alpha is None:
            factor = torch.empty(1).uniform_(0.5, 1.0).item()
        else:
            factor = torch.empty(1).uniform_(alpha, 1 + alpha).item()

        # Apply the contrast adjustment
        x_contrasted = torch.clamp(x_arr * factor, 0, 255).to(torch.uint8)

        if not is_batched:
            x_contrasted = x_contrasted[0]

        return x_contrasted

    def elastic_transform(
        self,
        img: torch.Tensor,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> torch.Tensor:
        """
        Apply elastic transformation to thermal images.
        
        Args:
            img: Input tensor in format [T, H, W, C] (PyTorch tensor)
            alpha: Scaling factor for displacement
            sigma: Standard deviation for Gaussian filter
            random_state: Random state for reproducibility
        
        Returns:
            Augmented tensor in same format
        """
        # Convert to numpy for scipy operations (scipy works better with numpy)
        arr = img.clone().cpu().numpy()
        is_batched = (arr.ndim == 4)
        
        if not is_batched:
            arr = np.expand_dims(arr, axis=0)
        
        T, h, w, c = arr.shape
        
        alpha = alpha if alpha is not None else self.image_size * 0.08
        sigma = sigma if sigma is not None else self.image_size * 0.08
        random_state = random_state or np.random.RandomState(None)

        # Generate displacement fields
        dx = (
            gaussian_filter(
                (random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.vstack([(y_coords + dy).ravel(), (x_coords + dx).ravel()])

        # Apply same deformation to all frames
        distorted = np.zeros_like(arr)
        for t in range(T):
            for ch in range(c):
                distorted[t, :, :, ch] = map_coordinates(
                    arr[t, :, :, ch], indices, order=1, mode="reflect"
                ).reshape(h, w)

        distorted = np.clip(distorted, 0, 255).astype(np.uint8)
        
        # Convert back to torch tensor
        distorted = torch.from_numpy(distorted)
        
        if not is_batched:
            distorted = distorted[0]
        
        return distorted


# --- Module-level convenience functions ---
_default_augmentor = ThermalAugmentor()


def occlusion(img, **kwargs):
    """Apply thermal erase augmentation."""
    return _default_augmentor.thermal_erase(img, **kwargs)


def horizontal_flip(images, prob=0.5, boxes=None):
    """Apply horizontal flip augmentation."""
    return _default_augmentor.horizontal_flip(images, prob, boxes)


def contrast(img, **kwargs):
    """Apply thermal contrast adjustment."""
    return _default_augmentor.thermal_contrast(img, **kwargs)


def brightness_contrast(img, **kwargs):
    """Apply brightness and contrast adjustment."""
    return _default_augmentor.brightness_contrast(img, **kwargs)


def elastic(img, **kwargs):
    """Apply elastic transformation."""
    return _default_augmentor.elastic_transform(img, **kwargs)

def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize with dimensions (C, T, H, W)
        mean (tuple): mean values (expects values in same range as tensor)
        std (tuple): std values (expects values in same range as tensor)
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    
    # Your mean/std are for [0,1] range, but tensor is [0,255]
    # Convert mean/std to [0,255] range
    if tensor.max() > 1.0:  # Tensor is in [0,255] range
        mean = torch.tensor(mean) * 255.0
        std = torch.tensor(std) * 255.0
    else:
        mean = torch.tensor(mean)
        std = torch.tensor(std)
    
    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)
    return tensor