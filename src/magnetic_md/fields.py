"""Magnetic field protocol functions for simulation control."""

from typing import Callable


def constant_field(field_on: bool) -> Callable[[float], bool]:
    """Create a constant field protocol.
    
    Args:
        field_on: Whether the field is ON (True) or OFF (False)
    
    Returns:
        A callable that takes time and returns the field state
    """
    def protocol(t: float) -> bool:
        return field_on
    return protocol


def square_wave(t_on: float, t_off: float, start_on: bool = False) -> Callable[[float], bool]:
    """Create a square wave field protocol with alternating ON/OFF periods.
    
    Args:
        t_on: Duration of ON period (seconds)
        t_off: Duration of OFF period (seconds)
        start_on: Whether to start with field ON (True) or OFF (False)
    
    Returns:
        A callable that takes time and returns the field state
    """
    def protocol(t: float) -> bool:
        period = t_on + t_off
        t_mod = t % period
        
        if start_on:
            # Start ON: [0, t_on) is ON, [t_on, period) is OFF
            return t_mod < t_on
        else:
            # Start OFF: [0, t_off) is OFF, [t_off, period) is ON
            return t_mod >= t_off
    
    return protocol


def delayed_activation(t_delay: float) -> Callable[[float], bool]:
    """Create a delayed activation protocol: OFF until t_delay, then ON.
    
    Args:
        t_delay: Time at which to turn field ON (seconds)
    
    Returns:
        A callable that takes time and returns the field state
    """
    def protocol(t: float) -> bool:
        return t >= t_delay
    
    return protocol
