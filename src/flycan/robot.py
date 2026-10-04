"""Replaceable movement boundary; watchdog is independent of model inference.
Drivers must execute quickly and be non-blocking. A remote driver must additionally
implement a watchdog on the robot itself: a host watchdog cannot stop a lost link.
"""
from dataclasses import dataclass, asdict
import math
import threading
import time
from typing import Protocol, Callable

@dataclass(frozen=True)
class MotionCommand:
    sequence: int
    action: str
    linear_mps: float
    angular_rps: float
    ttl_ms: int = 500
    def to_dict(self):return asdict(self)

class RobotDriver(Protocol):
    name: str
    simulated: bool
    def apply(self, command: MotionCommand) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...
    def close(self) -> None: ...

class MockDogDriver:
    name='Simulated Freenove adapter';simulated=True
    def __init__(self):
        self.x=self.y=self.heading=0.;self.linear=self.angular=0.;self.last=time.monotonic();self.commands=0
    def _integrate(self):
        now=time.monotonic();dt=min(now-self.last,.2);self.last=now
        self.heading+=self.angular*dt
        self.x+=self.linear*math.cos(self.heading)*dt;self.y+=self.linear*math.sin(self.heading)*dt
    def apply(self,command):
        self._integrate();self.linear=command.linear_mps;self.angular=command.angular_rps;self.commands+=1
    def stop(self):self._integrate();self.linear=self.angular=0.
    def status(self):
        self._integrate()
        return {'name':self.name,'simulated':True,'x':self.x,'y':self.y,'heading':self.heading,
                'linear_mps':self.linear,'angular_rps':self.angular,'commands':self.commands}
    def close(self):self.stop()

class CallbackDogDriver:
    """Plug in the user's vendor movement code. Angles are radians; +turn is left.
Callbacks receive physical setpoints. Convert them to vendor units in the adapter.
"""
    name='Freenove callback adapter';simulated=False
    def __init__(self, move: Callable[[float,float],None], stop: Callable[[],None]):
        self.move_callback=move;self.stop_callback=stop;self.last_command=None
    def apply(self,command):
        if command.action=='stop':self.stop()
        else:self.move_callback(command.linear_mps,command.angular_rps)
        self.last_command=command.to_dict()
    def stop(self):self.stop_callback()
    def status(self):return {'name':self.name,'simulated':False,'last_command':self.last_command}
    def close(self):self.stop()

class RobotBridge:
    def __init__(self,driver=None,clock=time.monotonic,watchdog=True):
        self.driver=driver or MockDogDriver();self.clock=clock;self.lock=threading.RLock()
        self.armed=False;self.estop=False;self.reason='Disarmed';self.sequence=0
        self.deadline=0.;self.command=MotionCommand(0,'stop',0.,0.)
        self.closed=threading.Event();self.thread=None
        self.driver.stop()
        if watchdog:
            self.thread=threading.Thread(target=self._watch,name='dog-watchdog',daemon=True);self.thread.start()
    def _halt(self,reason,disarm=False):
        self.reason=reason
        if disarm:self.armed=False
        self.sequence+=1;self.command=MotionCommand(self.sequence,'stop',0.,0.)
        self.driver.stop()
    def arm(self):
        with self.lock:
            if self.estop:raise ValueError('Clear stop latch before enabling movement')
            self.driver.stop();self.armed=True;self.deadline=self.clock()+.5;self.reason='Awaiting fresh input'
    def disarm(self):
        with self.lock:self._halt('Disarmed',True)
    def emergency_stop(self):
        with self.lock:self.estop=True;self._halt('Stop latched',True)
    def clear_stop(self):
        with self.lock:self.estop=False;self._halt('Disarmed',True)
    def submit(self,action,front_range_m,observed_at):
        with self.lock:
            now=self.clock()
            if self.estop:self._halt('Stop latched',True)
            elif not self.armed:self._halt('Disarmed')
            elif not math.isfinite(observed_at) or not 0<=now-observed_at<=.6:self._halt('Stale sensor input',True)
            elif not math.isfinite(front_range_m) or front_range_m<0:self._halt('Invalid range',True)
            elif front_range_m<.20:self._halt('Obstacle within 0.20 m')
            elif action not in ('forward','left','right','stop'):self._halt('Unknown action',True)
            else:
                linear=.08 if action=='forward' else 0.
                angular=.45 if action=='left' else (-.45 if action=='right' else 0.)
                self.sequence+=1;self.command=MotionCommand(self.sequence,action,linear,angular)
                try:self.driver.apply(self.command)
                except Exception:
                    self._halt('Driver error',True);raise
                self.reason='Policy command' if action!='stop' else 'Policy stop'
            self.deadline=now+.5
            return self.status()
    def check_timeout(self):
        with self.lock:
            if self.armed and self.clock()>self.deadline:self._halt('Command timeout',True)
    def _watch(self):
        while not self.closed.wait(.05):
            try:self.check_timeout()
            except Exception:
                # An exception never silently leaves the bridge armed.
                with self.lock:self.armed=False;self.reason='Driver stop failed'
    def status(self):
        with self.lock:
            return {'armed':self.armed,'estop':self.estop,'reason':self.reason,
                    'command':self.command.to_dict(),'driver':self.driver.status()}
    def close(self):
        self.closed.set()
        if self.thread:self.thread.join(timeout=1)
        with self.lock:self._halt('Shutdown',True);self.driver.close()
