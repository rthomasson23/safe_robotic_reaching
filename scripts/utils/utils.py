import time

class sleeper(object):
  def __init__(self, p, timeStep):
    self._p = p
    self.timeStep = timeStep

  def sleepSim(self, sleepTime, stepFuncs=None):
    """ Step through the simluation for some amount of time. """
    steps = int(round(sleepTime / self.timeStep))
    # step once first to place the gripper in place
    for i in range(steps):
        if (stepFuncs != None):
          stepFuncs()
        self._p.stepSimulation()
        time.sleep(self.timeStep)
