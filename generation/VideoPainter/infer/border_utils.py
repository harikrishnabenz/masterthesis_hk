"""
Enhanced video border handling to prevent black artifacts during camera movement.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def create_border_aware_mask(
    frame: np.ndarray,
    binary_mask: np.ndarray,
    method: str = "inpaint"
) -> np.ndarray:
    """
    Create a masked frame with better border handling to avoid black artifacts.
    
    Args:
        frame: Original frame as numpy array (H, W, 3)
        binary_mask: Binary mask as numpy array (H, W)
        method: Method for handling masked regions ["inpaint", "blur", "interpolate"]
    
    Returns:
        Masked frame with better border handling
    """
    if method == "inpaint":
        # Use OpenCV's inpainting for content-aware filling
        mask_dilated = cv2.dilate(binary_mask.astype(np.uint8), 
                                 np.ones((3, 3), np.uint8), 
                                 iterations=1)
        result = cv2.inpaint(frame, mask_dilated, 3, cv2.INPAINT_TELEA)
    
    elif method == "blur":
        # Use Gaussian blur of surrounding regions
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        result = np.where(binary_mask_expanded, blurred_frame, frame)
    
    elif method == "interpolate":
        # Use nearest neighbor interpolation of border pixels
        result = frame.copy()
        mask_coords = np.where(binary_mask)
        
        if len(mask_coords[0]) > 0:
            # Find nearest valid pixels for each masked pixel
            for i, j in zip(mask_coords[0], mask_coords[1]):
                # Search in expanding squares until we find a valid pixel
                for radius in range(1, min(frame.shape[:2]) // 2):
                    y_min = max(0, i - radius)
                    y_max = min(frame.shape[0], i + radius + 1)
                    x_min = max(0, j - radius)
                    x_max = min(frame.shape[1], j + radius + 1)
                    
                    # Get border pixels of this square
                    border_mask = np.zeros_like(binary_mask)
                    border_mask[y_min:y_max, x_min:x_max] = 1
                    border_mask[y_min+1:y_max-1, x_min+1:x_max-1] = 0
                    
                    # Find valid (non-masked) border pixels
                    valid_border = border_mask & (~binary_mask)
                    valid_coords = np.where(valid_border)
                    
                    if len(valid_coords[0]) > 0:
                        # Use mean of valid border pixels
                        result[i, j] = np.mean(frame[valid_coords], axis=0)
                        break
    else:
        # Fallback to black if method unknown
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        result = np.where(binary_mask_expanded, 0, frame)
    
    return result


def read_video_with_improved_mask(
    video_path: str, 
    masks: np.ndarray, 
    mask_id: int, 
    skip_frames_start: int = 0, 
    skip_frames_end: int = -1, 
    mask_background: bool = False, 
    fps: int = 0,
    border_method: str = "inpaint"
) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], int]:
    """
    Enhanced version of read_video_with_mask with better border handling.
    """
    from diffusers.utils import load_video
    
    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = masks[skip_frames_start:skip_frames_end]
    
    # Read fps
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    masked_video = []
    binary_masks = []
    
    for frame, frame_mask in zip(video, mask):
        frame_array = np.array(frame)
        
        binary_mask = (frame_mask == mask_id)
        
        # Use improved border handling instead of pure black
        masked_frame_array = create_border_aware_mask(
            frame_array, binary_mask, method=border_method
        )
        
        masked_video.append(Image.fromarray(masked_frame_array.astype(np.uint8)).convert("RGB"))
        
        if mask_background:
            binary_mask_image = np.where(binary_mask, 0, 255).astype(np.uint8)
        else:
            binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))
    
    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps


def apply_border_padding(
    frames: List[np.ndarray], 
    padding_percent: float = 0.1
) -> List[np.ndarray]:
    """
    Apply border padding to frames to provide extra content during camera movement.
    
    Args:
        frames: List of video frames as numpy arrays
        padding_percent: Percentage of frame dimensions to add as padding (0.1 = 10%)
    
    Returns:
        List of padded frames
    """
    if not frames:
        return frames
        
    h, w = frames[0].shape[:2]
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)
    
    padded_frames = []
    for frame in frames:
        # Reflect padding to avoid harsh borders
        padded = cv2.copyMakeBorder(
            frame, pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_REFLECT_101
        )
        padded_frames.append(padded)
    
    return padded_frames