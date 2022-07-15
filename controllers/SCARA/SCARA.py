import math

import numpy

import controller

offset = 0.1
l1 = 0.475
l2 = 0.4
tol = 0.1
dh = numpy.array([
    [None, 0, 0, l1],
    [None, 0, 0, l2],
    [0.5 * numpy.pi, None, 0, 0],
    [None, 0, numpy.pi, 0]
])


def rot2eul(R):
    beta = -numpy.arcsin(R[2, 0])
    alpha = numpy.arctan2(R[2, 1] / numpy.cos(beta), R[2, 2] / numpy.cos(beta))
    gamma = numpy.arctan2(R[1, 0] / numpy.cos(beta), R[0, 0] / numpy.cos(beta))
    return numpy.array((alpha, beta, gamma))


def get_transform_matrix(M):
    return numpy.reshape(M, (4, 4))


def decompose(T):
    return T[:3, :3], T[:3, 3]


def get_joint_transform_matrix(q, n, dh, joint_types):
    T = numpy.eye(4)
    for i in range(n):
        theta, d, alpha, a = dh[i]
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        cos_alpha = numpy.cos(alpha)
        sin_alpha = numpy.cos(alpha)
        T = numpy.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ])

    return T


def jacobian(q, joint_types, pe):
    J = numpy.zeros((6, len(q)))

    for i, joint_type in enumerate(joint_types):
        M = get_joint_transform_matrix(q, i, dh, joint_types)
        R, T = decompose(M)
        Zi = R @ numpy.array([[0, 0, 1]]).T

        Jp, Jo = numpy.zeros(3), numpy.zeros(3)
        if joint_type == 'p':
            Jp = Zi
            Jo = numpy.zeros((3, 1))
        elif joint_type == 'r':
            Jp = numpy.cross(Zi.T, pe - T).reshape((3, 1))
            Jo = Zi

        J[:, i] = numpy.vstack([Jp, Jo]).flatten()

    return J


sim = controller.Supervisor()
timestep = int(sim.getBasicTimeStep())
goal = sim.getFromDef('rubber_duck')
effector = sim.getFromDef('suction_pad')
motors = [sim.getDevice(f'joint{i}_motor') for i in range(1, 5)]
sensors = [motor.getPositionSensor() for motor in motors]
for sensor in sensors:
    sensor.enable(timestep)

q = numpy.array([sensor.getValue() for sensor in sensors])
min_q = numpy.array([motor.getMinPosition() for motor in motors])
max_q = numpy.array([motor.getMaxPosition() for motor in motors])

effector_rotation_matrix, effector_translation = decompose(get_transform_matrix(effector.getPose()))
effector_pose = numpy.hstack([effector_translation.reshape(1, 3), rot2eul(effector_rotation_matrix).reshape(1, 3)])
goal_rotation_matrix, goal_translation = decompose(get_transform_matrix(goal.getPose()))
goal_pose = numpy.hstack([goal_translation.reshape(1, 3), rot2eul(goal_rotation_matrix).reshape(1, 3)])
distance = numpy.linalg.norm(goal_pose - effector_pose)

while distance > tol and sim.step(timestep) != -1:
    dh[0][0] = q[0]
    dh[1][0] = q[1]
    dh[2][1] = q[2]
    dh[3][0] = q[3] + numpy.pi

    J = jacobian(q, 'rrpr', effector_translation.flatten())
    q_dot = numpy.linalg.pinv(J) @ (goal_pose - effector_pose).T
    q += q_dot.flatten() * timestep / 1000

    for i, motor in enumerate(motors):
        if (min_q[i] != 0 and max_q[i] != 0):
            q[i] = min(max(q[i], min_q[i]), max_q[i])
        motor.setPosition(q[i])

    q = numpy.array([sensor.getValue() for sensor in sensors])

    # if sim.step(timestep) == -1:
    #     break

    effector_rotation_matrix, effector_translation = decompose(get_transform_matrix(effector.getPose()))
    effector_pose = numpy.hstack([effector_translation.reshape(1, 3), rot2eul(effector_rotation_matrix).reshape(1, 3)])
    goal_rotation_matrix, goal_translation = decompose(get_transform_matrix(goal.getPose()))
    goal_pose = numpy.hstack([goal_translation.reshape(1, 3), rot2eul(goal_rotation_matrix).reshape(1, 3)])
    distance = numpy.linalg.norm(goal_pose - effector_pose)
