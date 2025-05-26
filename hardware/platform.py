import os
import platform

def is_raspberry_pi():
    """Check if we're running on a Raspberry Pi"""
    return platform.machine() in ('armv7l', 'aarch64')

def get_gpio_module():
    """Get the appropriate GPIO module based on platform"""
    if is_raspberry_pi():
        import RPi.GPIO as GPIO
        return GPIO
    else:
        from .mock_gpio import GPIO
        return GPIO