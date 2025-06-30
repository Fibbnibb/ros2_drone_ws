#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# For sending commands to PX4
from px4_msgs.msg import VehicleCommand
# For reading the vehicle status
from px4_msgs.msg import VehicleStatus

class TakeoffNode(Node):

    def __init__(self):
        super().__init__('takeoff_node')

        # 1) Subscribe to vehicle status (VehicleStatus msg on /fmu/vehicle_status/out)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            10
        )

        # 2) Publisher for sending VehicleCommand to PX4
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )

        # Timed event for sending commands
        self.timer = self.create_timer(2.0, self.send_takeoff_commands)
        self.sent_arming_cmd = False
        self.sent_takeoff_cmd = False

        self.get_logger().info("TakeoffNode started. Will send ARM and TAKEOFF commands.")

    def vehicle_status_callback(self, msg):
        # Print out relevant fields from the VehicleStatus
        # For example, arming_state or nav_state
        self.get_logger().info(f"Vehicle Status: arming_state={msg.arming_state}, nav_state={msg.nav_state}")

    def send_takeoff_commands(self):
        # 1) ARM once
        if not self.sent_arming_cmd:
            self.arm()
            self.sent_arming_cmd = True
            return

        # 2) Then issue TAKEOFF command
        if not self.sent_takeoff_cmd:
            self.takeoff()
            self.sent_takeoff_cmd = True
            self.get_logger().info("TAKEOFF command sent.")
            return

        self.get_logger().info("Commands sent. Waiting...")

    def arm(self):
        cmd = VehicleCommand()
        cmd.param1 = 1.0  # 1 = arm, 0 = disarm
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True

        self.cmd_pub.publish(cmd)
        self.get_logger().info("ARM command published.")

    def takeoff(self):
        cmd = VehicleCommand()
        cmd.param1 = 0.0  # Minimum pitch
        cmd.param2 = 0.0
        cmd.param3 = 0.0
        cmd.param4 = 0.0  # Yaw
        cmd.param5 = 0.0  # Latitude
        cmd.param6 = 0.0  # Longitude
        cmd.param7 = 5.0  # Altitude (meters)
        cmd.command = VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TakeoffNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
