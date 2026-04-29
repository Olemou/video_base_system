import torch
import torchvision
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple

class MotionAugmentation:
    """Motion augmentation for creating fake video clips from single images"""
    
    def __init__(self, fpc: int = 16, motion_type: str = 'mixed', 
                 motion_intensity: str = 'medium', seed: Optional[int] = None):
        self.fpc = fpc
        self.motion_type = motion_type
        self.motion_intensity = motion_intensity
        if seed is not None:
            np.random.seed(seed)
        
        # Set motion intensity ranges (in pixels)
        self.intensity_ranges = {
            'micro': {'x': 5, 'y': 5},
            'small': {'x': 15, 'y': 15},
            'medium': {'x': 30, 'y': 25},
            'large': {'x': 60, 'y': 50}
        }
    
    def generate_linear_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linear motion with easing"""
        t = np.linspace(0, 1, self.fpc)
        intensity = self.intensity_ranges[self.motion_intensity]
        motion_x = t * intensity['x']
        motion_y = t * intensity['y']
        return motion_x, motion_y
    
    def generate_random_walk_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random walk with smoothing"""
        intensity = self.intensity_ranges[self.motion_intensity]
        step_scale = intensity['x'] / 8
        
        raw_x = np.cumsum(np.random.randn(self.fpc) * step_scale)
        raw_y = np.cumsum(np.random.randn(self.fpc) * step_scale)
        
        sigma = max(1, self.fpc // 8)
        motion_x = gaussian_filter1d(raw_x, sigma=sigma)
        motion_y = gaussian_filter1d(raw_y, sigma=sigma)
        
        motion_x = np.clip(motion_x, -intensity['x'], intensity['x'])
        motion_y = np.clip(motion_y, -intensity['y'], intensity['y'])
        
        return motion_x, motion_y
    
    def generate_circular_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Circular/elliptical motion"""
        t = np.linspace(0, 2 * np.pi, self.fpc)
        intensity = self.intensity_ranges[self.motion_intensity]
        motion_x = np.sin(t) * intensity['x'] * 0.8
        motion_y = np.cos(t * 1.3) * intensity['y'] * 0.7
        return motion_x, motion_y
    
    def generate_handheld_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Realistic handheld camera motion"""
        intensity = self.intensity_ranges[self.motion_intensity]
        
        # High frequency shake
        shake_x = np.random.randn(self.fpc) * (intensity['x'] * 0.15)
        shake_y = np.random.randn(self.fpc) * (intensity['y'] * 0.15)
        
        # Low frequency drift
        drift_steps_x = np.random.randn(self.fpc) * (intensity['x'] * 0.08)
        drift_steps_y = np.random.randn(self.fpc) * (intensity['y'] * 0.08)
        drift_x = np.cumsum(drift_steps_x)
        drift_y = np.cumsum(drift_steps_y)
        
        sigma = max(2, self.fpc // 6)
        drift_x = gaussian_filter1d(drift_x, sigma=sigma)
        drift_y = gaussian_filter1d(drift_y, sigma=sigma)
        
        motion_x = shake_x + drift_x
        motion_y = shake_y + drift_y
        
        motion_x = np.clip(motion_x, -intensity['x'], intensity['x'])
        motion_y = np.clip(motion_y, -intensity['y'], intensity['y'])
        
        return motion_x, motion_y
    
    def generate_mixed_motion(self, clip_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Mix motion types based on clip index (40/30/20/10 distribution)"""
        pattern_choice = clip_index % 10
        
        if pattern_choice < 4:  # 40% Handheld
            return self.generate_handheld_motion()
        elif pattern_choice < 7:  # 30% Random walk
            return self.generate_random_walk_motion()
        elif pattern_choice < 9:  # 20% Linear
            return self.generate_linear_motion()
        else:  # 10% Static
            return np.zeros(self.fpc), np.zeros(self.fpc)
    
    def generate_static_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        """No motion"""
        return np.zeros(self.fpc), np.zeros(self.fpc)
    
    def get_motion(self, clip_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get motion trajectory based on motion_type"""
        if self.motion_type == 'linear':
            return self.generate_linear_motion()
        elif self.motion_type == 'random_walk':
            return self.generate_random_walk_motion()
        elif self.motion_type == 'circular':
            return self.generate_circular_motion()
        elif self.motion_type == 'handheld':
            return self.generate_handheld_motion()
        elif self.motion_type == 'mixed':
            return self.generate_mixed_motion(clip_index)
        else:  # static
            return self.generate_static_motion()
    
    def apply_motion(self, buffer: torch.Tensor, clip_index: int = 0) -> torch.Tensor:
        """Apply motion to buffer"""
        fpc = buffer.shape[0]
        motion_x, motion_y = self.get_motion(clip_index)
        
        augmented_frames = []
        for i in range(fpc):
            shift_x = int(motion_x[i])
            shift_y = int(motion_y[i])
            # Apply cyclic shift
            shifted_frame = torch.roll(buffer[i], shifts=(shift_y, shift_x), dims=(0, 1))
            augmented_frames.append(shifted_frame)
        
        return torch.stack(augmented_frames)