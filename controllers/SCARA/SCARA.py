from functools import reduce

import numpy
from numpy.linalg import pinv
from numpy.linalg import norm

from controller import Supervisor


L_1 = 0.475
L_2 = 0.4
OFFSET = 0.1
c = lambda theta: numpy.cos(theta)
s = lambda theta: numpy.sin(theta)
pi = numpy.pi
pi_2 = 0.5 * numpy.pi
alpha = 0.01
tol = 0.01


def dh(theta_1, theta_2, theta_3, d_4): 
    return numpy.array([
        [       theta_1,   OFFSET,  0, L_1],
        [       theta_2,        0,  0, L_2],
        [pi_2 + theta_3,        0, pi,   0],
        [          pi_2,      d_4,  0,   0],
    ])


def T(theta, d, alpha, a): 
    return numpy.array([
        [           c(theta),           -s(theta),         0,             a],
        [s(theta) * c(alpha), c(theta) * c(alpha), -s(alpha), -s(alpha) * d],
        [s(theta) * s(alpha), c(theta) * s(alpha),  c(alpha),  c(alpha) * d],
        [                  0,                   0,         0,             1],
    ])


def fkine(n, *args):
    return reduce(numpy.matmul, [T(*p) for p in dh(*args)[:n]])


def jacobian(q):
    """
    s1, s2, _, s3 = numpy.sin(q)
    c1, c2, _, c3 = numpy.cos(q)
    return numpy.array([
        [-L1*s1 - L2*s1*s2, -L2*s1*s2,  0,  0],
        [ L1*c1 + L2*c1*c2,  L2*c1*c2,  0,  0],
        [                0,         0, -1,  0],
        [                0,         0,  0,  0],
        [                0,         0,  0,  0],
        [                1,         1,  0, -1],
    ])
    """
    z0 = numpy.array([0, 0, 1]).reshape((3, 1))
    
    pi = [fkine(i, *q)[:-1, 3].reshape((3, 1)) for i in range(1, 5)]
    pe = fkine(None, *q)[:-1, 3].reshape((3, 1))
    
    jpi = [numpy.cross(numpy.array([0, 0, 1]), (p - pe).ravel()).reshape((3, 1)) for p in pi]
     
    return numpy.hstack([
        numpy.vstack([jpi[0], z0]),
        numpy.vstack([jpi[1], z0]),
        numpy.array([[0, 0, -1, 0, 0, 0]]).T,
        numpy.vstack([jpi[3], z0]),  
    ])


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

deviceNames = []
for i in range(robot.getNumberOfDevices()):
    deviceNames.append(robot.getDeviceByIndex(i).getName())

numberOfScrews = 0
motors = []
sensors = []
end_effector_gps = robot.getDevice('end_effector_gps')
end_effector_gps.enable(timestep)
rubber_duck = robot.getFromDef('rubber_duck')

for i in range(1, robot.getNumberOfDevices()):
    linearMotorName = f'joint{i}_motor'
    positionSensorName = f'joint{i}_sensor'
    if linearMotorName in deviceNames and positionSensorName in deviceNames:
        motors.append(robot.getDevice(linearMotorName))
        sensors.append(robot.getDevice(positionSensorName))
    else:
        break

for sensor in sensors:
    sensor.enable(timestep)

distance = 1000.0
while robot.step(timestep) != -1 and distance > tol:
    q = numpy.array([sensor.getValue() for sensor in sensors])
    J = jacobian(q)
    J_cross = pinv(J)
    
    s1, s2, _, s3 = numpy.sin(q)
    c1, c2, _, c3 = numpy.cos(q)
    d = q[2]
    e_pos = fkine(None, *q) @ numpy.array([[0], [0], [1], [1]])
    e_ori = numpy.array([[0], [0], [q[3]]])
    current_e = numpy.array([*e_pos[:-1], *e_ori]).T
    rubber_duck_position = rubber_duck.getField('translation').getSFVec3f()
    goal_e = numpy.array([[*rubber_duck_position, 0, 0, 0]])
    delta_e = goal_e - current_e
    delta_q = J_cross @ delta_e.T 
    for i in range(len(q)):
        q[i] += alpha * delta_q[i]
    q[2] = 0
    for motor, position in zip(motors, q):
        motor.setPosition(position)

    distance = norm(delta_e)