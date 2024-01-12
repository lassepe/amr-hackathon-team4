#!/usr/bin/env python

import threading
import rospy
import math
import tf

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped


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


class JackalControl:
    def __init__(self, dt=0.1, default_control_action=[0.0, 0.0]):
        self.default_control_action = default_control_action
        self.latest_strategy = GuardedVariable(None)
        self.latest_odom = GuardedVariable(None)
        self.latest_goal = GuardedVariable(None)
        self.current_control_action = JackalControl.update_action(
            Twist(), default_control_action
        )

        self.odometry_subscriber = rospy.Subscriber(
            "/robot_ekf/odometry", Odometry, self.odometry_callback
        )

        self.goal_subscriber = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_callback
        )

        self.control_action_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.dt = dt
        self.strategy_updater_thread = threading.Thread(
            target=self.strategy_update_task
        )
        # TODO: could also warm-up the julia solver here

    def goal_callback(self, msg):
        """
        Update the latest goal.
        """
        self.latest_goal.set(msg)

    def odometry_callback(self, msg):
        self.latest_odom.set(msg)

    def update_action(twist_action, action):
        """
        Update the test action with the given action.
        """
        twist_action.linear.x = action[0]
        twist_action.angular.z = action[1]
        return twist_action

    def strategy_update_task(self):
        """
        Continuously compute the control action to be taken by the robot from the current
        state estimate.
        """
        rate = rospy.Rate(2 / self.dt)
        while not rospy.is_shutdown():
            odom = self.latest_odom.get()
            odom = self.latest_odom.get()
            goal = self.latest_goal.get()
            if odom and goal:
                time_stamp = odom.header.stamp
                new_strategy = self.compute_strategy(odom, goal)
                self.latest_strategy.set(
                    {"time_stamp": time_stamp, "controls": new_strategy["us"]}
                )
            rate.sleep()

    def compute_strategy(self, odom, goal):
        """
        Compute a receding-horizon strategy for the robot to follow.
        """

        # simple controller: choose the turn-rate based on the error to the relative goal position,
        # and move forward at a constant rate.

        goal_x = goal.pose.position.x - odom.pose.pose.position.x
        goal_y = goal.pose.position.y - odom.pose.pose.position.y
        relative_goal_angle = math.atan2(goal_y, goal_x)

        # turn rate proportional to the angle error
        ego_z_angle = tf.transformations.euler_from_quaternion(
            (
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            )
        )[2]
        turn_rate = compute_angle_error(relative_goal_angle, ego_z_angle)
        # acceleration proportional to the distance to the goal
        distance_to_goal = math.sqrt(goal_x**2 + goal_y**2)
        velocity_clipped = min(0.5, distance_to_goal)
        assert velocity_clipped >= 0.0

        return {
            "us": [
                [velocity_clipped, turn_rate],
            ]
        }

    def publish_control_action(self, control_action):
        self.current_control_action = JackalControl.update_action(
            self.current_control_action, control_action
        )
        self.control_action_publisher.publish(self.current_control_action)

    def run(self):
        """
        Play out the control actions from the current strategy at a fixed rate.
        Advance the index into the strategy array at each step.
        """
        self.strategy_updater_thread.start()
        rate = rospy.Rate(1.0 / self.dt)
        strategy = None
        horizon_index = 0
        while not rospy.is_shutdown():
            latest_strategy = self.latest_strategy.get()
            if latest_strategy is not strategy:
                strategy = latest_strategy
                horizon_index = 0
            if strategy is not None:
                controls = strategy["controls"]
                if horizon_index < len(controls):
                    control_action = controls[horizon_index]
                    self.publish_control_action(control_action)
                    horizon_index += 1
                else:
                    rospy.loginfo("Strategy exhausted.")
                    self.publish_control_action(self.default_control_action)
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("jackal_control_node", anonymous=True)
        jackal_control = JackalControl()
        jackal_control.run()
    except rospy.ROSInterruptException:
        pass
