#!/usr/bin/env python

import threading
import rospy
import math
import utils

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from julia_trajectory_optimizer import JuliaTrajectoryOptimizer


class JackalControl:
    def __init__(self, dt=0.1):
        self.latest_strategy = utils.GuardedVariable(None)
        self.latest_state = utils.GuardedVariable(None)
        self.latest_goal = utils.GuardedVariable(None)
        self.latest_obstacle = utils.GuardedVariable(None)

        self.odometry_subscriber = rospy.Subscriber(
            "/robot_ekf/odometry", Odometry, self.odometry_callback
        )

        self.goal_subscriber = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_callback
        )

        self.obstacle_subscriber = rospy.Subscriber(
            "/vicon/obstacle1", PoseWithCovarianceStamped, self.obstacle_callback
        )

        self.control_action_publisher = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10
        )
        self.dt = dt
        self.ros_communication_thread = threading.Thread(
            target=self.ros_communication_task
        )

        self.strategy_update_thread = threading.Thread(
            target=self.strategy_update_task)

    def goal_callback(self, msg):
        """
        Update the latest goal.
        """
        # TODO: also extract orientation
        goal_orientation = utils.get_z_angle_from_quaternion(
            msg.pose.orientation)
        goal = [msg.pose.position.x, msg.pose.position.y, goal_orientation]
        self.latest_goal.set(goal)

    def odometry_callback(self, msg):
        state = utils.vector_from_odom(msg)
        self.latest_state.set(state)

    def obstacle_callback(self, msg):
        obstacle_orientation = utils.get_z_angle_from_quaternion(
            msg.pose.pose.orientation
        )
        obstacle = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            obstacle_orientation,
        ]
        self.latest_obstacle.set(obstacle)

    def strategy_update_task(self):
        """
        Continuously compute the control action to be taken by the robot from the current
        state estimate.
        """

        motion_controller = JuliaTrajectoryOptimizer()

        rate = rospy.Rate(2 / self.dt)
        while not rospy.is_shutdown():
            state = self.latest_state.get()
            goal = self.latest_goal.get()
            obstacle = self.latest_obstacle.get()
            if state and goal and obstacle:
                # start_time = rospy.get_time()
                new_strategy = motion_controller.compute_strategy(
                    state, goal, obstacle)
                # end_time = rospy.get_time()
                # print("Time taken: {}".format(end_time - start_time))
                self.latest_strategy.set(new_strategy)
            rate.sleep()

    def ros_communication_task(self):
        """
        Play out the control actions from the current strategy at a fixed rate.
        Advance the index into the strategy array at each step.
        """
        rate = rospy.Rate(1 / self.dt)
        strategy = None
        while not rospy.is_shutdown():
            strategy = self.latest_strategy.get()
            state = self.latest_state.get()

            if strategy:
                assert state is not None
                control_action = strategy(state)
                self.publish_control_action(control_action)
            rate.sleep()

    def publish_control_action(self, control_action):
        twist_msg = Twist()
        twist_msg.linear.x = control_action[0]
        twist_msg.angular.z = control_action[1]
        self.control_action_publisher.publish(twist_msg)

    def run(self):
        """
        Play out the control actions from the current strategy at a fixed rate.
        Advance the index into the strategy array at each step.
        """
        self.ros_communication_thread.start()
        self.strategy_update_thread.start()


if __name__ == "__main__":
    try:
        rospy.init_node("jackal_control_node", anonymous=True)
        jackal_control = JackalControl()
        jackal_control.run()
    except rospy.ROSInterruptException:
        pass
