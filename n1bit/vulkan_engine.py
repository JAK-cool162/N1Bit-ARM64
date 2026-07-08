import os
import sys

# Try to import torch to check Vulkan availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def is_vulkan_available() -> bool:
    """
    Checks if Vulkan device acceleration is supported by the PyTorch backend
    on the current PC or phone (Termux) system.
    """
    if not HAS_TORCH:
        return False
        
    try:
        # PyTorch check for Vulkan support
        return torch.is_vulkan_available()
    except AttributeError:
        return False

def get_vulkan_device():
    """
    Returns PyTorch Vulkan device if available, otherwise falls back.
    """
    if is_vulkan_available():
        print("[Vulkan] Native Vulkan compute device detected! Accelerating 1-bit operations on GPU.")
        return torch.device("vulkan")
    return None

def get_termux_vulkan_guide() -> str:
    """
    Returns a step-by-step setup guide for enabling Vulkan GPU acceleration 
    inside Termux on Android devices.
    """
    return """
============================================================
       TERMUX VULKAN GPU ACCELERATION SETUP GUIDE
============================================================
To enable Vulkan compute shaders for 1-bit LLMs on your mobile GPU:

1. Update Termux packages:
   $ pkg update && pkg upgrade -y

2. Install Android Vulkan loader and Mesa drivers:
   $ pkg install mesa-vulkan libvulkan -y

3. Verify Vulkan is active on your phone's GPU:
   $ pkg install vulkan-tools -y
   $ vulkaninfo

4. Once Vulkan is active, our 1-bit PyTorch backend will 
   automatically detect the GPU and route all matrix operations 
   through Vulkan compute shaders!
============================================================
"""
