import threading
import math
import tf


class GuardedVariable:
    def __init__(self, value):
        self.lock = threading.Lock()
        self.value = value

    def get(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = value


class OpenLoopStrategy:
    def __init__(self, input_sequence, default_control_action=[0.0, 0.0]):
        self.input_sequence = input_sequence
        self.current_index = 0
        self.default_control_action = default_control_action

    def __call__(self, state):
        if self.current_index < len(self.input_sequence):
            control_action = self.input_sequence[self.current_index]
        else:
            print("strategy exhausted: using default control action")
            control_action = self.default_control_action
        self.current_index += 1

        return control_action


def wrap_angle(angle):
    """
    Wrap angles to -pi, pi.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_angle_error(angle1, angle2):
    """
    Compute the signed arc length between two angles
    """
    delta = wrap_angle(angle1) - wrap_angle(angle2)
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi
    return delta


def vector_from_odom(odom):
    ## turn rate proportional to the angle error
    ego_z_angle = tf.transformations.euler_from_quaternion(
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )
    )[2]

    ego_velocity = odom.twist.twist.linear.x

    return [
        odom.pose.pose.position.x,
        odom.pose.pose.position.y,
        ego_velocity,
        ego_z_angle,
    ]
