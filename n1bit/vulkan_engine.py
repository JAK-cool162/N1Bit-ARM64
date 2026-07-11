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
        return torch.is_vulkan_available()
    except AttributeError:
        return False

def get_vulkan_device():
    """Returns PyTorch Vulkan device if available, otherwise falls back."""
    if is_vulkan_available():
        print("[Vulkan] Native Vulkan compute device detected! Accelerating 1-bit operations on GPU.")
        return torch.device("vulkan")
    return None

def get_termux_vulkan_guide() -> str:
    """
    Returns a step-by-step setup guide for enabling Vulkan GPU acceleration 
    inside Termux on Android devices, resolving 'Unable to locate package' errors.
    """
    return """
============================================================
       TERMUX VULKAN GPU ACCELERATION SETUP GUIDE
============================================================
In standard Termux, GPU and Vulkan packages are not in the main repository,
which leads to 'Unable to locate package' errors.

To enable Vulkan compute on your phone's GPU, follow these steps:

1. Enable the Termux User Repo (TUR) and X11 repo:
   $ pkg install tur-repo x11-repo -y

2. Update your package lists completely:
   $ pkg update && pkg upgrade -y

3. Install the Android Vulkan loader bridge, Vulkan tools, and shader compilers:
   $ pkg install vulkan-loader-android vulkan-tools vulkan-headers shaderc -y

4. Verify Vulkan is active on your phone's Adreno/Mali GPU:
   $ vulkaninfo

5. Once active, our 1-bit backend will automatically query 
   Vulkan availability and accelerate matrices on GPU!
============================================================
"""
