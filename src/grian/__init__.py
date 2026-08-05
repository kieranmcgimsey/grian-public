"""grian — NEM electricity price forecasting and battery-dispatch simulation.

Models forecast day-ahead prices; a walk-forward MPC dispatcher trades a battery
against them, scored against a perfect-foresight oracle. The pieces:
``models`` (forecast), ``dispatch`` (trade), and ``evaluation`` (score).
"""

__version__ = "0.1.0"
