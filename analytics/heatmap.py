"""Spatial Traffic Density and Dwell Movement Heatmap Generator."""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class HeatmapGenerator:
    """Accumulates target spatial footprints to generate density & dwell thermal maps."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720, decay_factor: float = 0.999, kernel_size: int = 31):
        self.width = frame_width
        self.height = frame_height
        self.decay_factor = decay_factor
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.accum = np.zeros((self.height, self.width), dtype=np.float32)

    def reset(self):
        """Clears accumulated heatmap data."""
        self.accum = np.zeros((self.height, self.width), dtype=np.float32)

    def resize_if_needed(self, width: int, height: int):
        """Dynamically adapts heatmap matrix to incoming frame resolution."""
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.accum = cv2.resize(self.accum, (width, height), interpolation=cv2.INTER_LINEAR)

    def add_points(self, points: List[Tuple[int, int]], weight: float = 1.0):
        """
        Adds a batch of target foot points to the density accumulator.
        """
        if not points:
            return

        # Optional decay over time to prioritize recent activity
        if self.decay_factor < 1.0:
            self.accum *= self.decay_factor

        temp = np.zeros((self.height, self.width), dtype=np.float32)
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                temp[y, x] += weight

        # Apply Gaussian blur kernel to spread density
        blurred = cv2.GaussianBlur(temp, (self.kernel_size, self.kernel_size), 0)
        self.accum += blurred

    def generate_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET,
        threshold_cutoff: float = 0.05
    ) -> np.ndarray:
        """
        Renders a thermal heatmap blended seamlessly onto the video frame.
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        self.resize_if_needed(w, h)

        max_val = np.max(self.accum)
        if max_val <= 0:
            return frame.copy()

        # Normalize accumulator to 0 - 255
        norm_accum = np.clip(self.accum / max_val * 255.0, 0, 255).astype(np.uint8)

        # Apply false-color heatmap
        color_heat = cv2.applyColorMap(norm_accum, colormap)

        # Create threshold mask so cold (zero activity) areas remain transparent
        mask = norm_accum > int(threshold_cutoff * 255)

        output = frame.copy()
        output[mask] = cv2.addWeighted(frame, 1.0 - alpha, color_heat, alpha, 0)[mask]

        return output
