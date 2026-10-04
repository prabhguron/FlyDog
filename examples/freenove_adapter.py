"""Boundary for the movement code you will provide. No servo protocol is guessed.
Run this example for console-only movement calls:
  python examples/freenove_adapter.py
"""
from flycan.robot import CallbackDogDriver, RobotBridge
import time

def make_freenove_driver(forward, left, right, stop):
    """Callbacks must be non-blocking and accept calibrated speeds in SI units.
    forward(metres_per_second), left(radians_per_second), right(radians_per_second).
    stop() must stop ongoing gait movement, not merely enqueue a stop behind it.
    """
    def move(linear_mps,angular_rps):
        if linear_mps>0:forward(linear_mps)
        elif angular_rps>0:left(angular_rps)
        elif angular_rps<0:right(-angular_rps)
        else:stop()
    return CallbackDogDriver(move,stop)

if __name__=='__main__':
    # These logging callbacks cannot move hardware. Replace only after calibration.
    driver=make_freenove_driver(
        forward=lambda speed:print(f'forward: {speed:.2f} m/s'),
        left=lambda speed:print(f'left: {speed:.2f} rad/s'),
        right=lambda speed:print(f'right: {speed:.2f} rad/s'),
        stop=lambda:print('stop'))
    bridge=RobotBridge(driver)
    try:
        bridge.arm()
        bridge.submit('left',front_range_m=1.5,observed_at=time.monotonic())
        # In production submit fresh camera/range results from your sensor loop.
    finally:bridge.close()
