#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus
)


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile
        )

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        # Desired alt and forward distance in NED frame (z negative down)
        self.altitude_m = -7.62  # 25 ft above ground
        self.forward_dist_m = 1.52  # 5 ft

        # A simple state machine:
        # 0 = idle/wait, 1 = taking off, 2 = move forward, 3 = land
        self.state = 0

        # Create a timer to publish control commands at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # 90 degrees in radians
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints: x={x}, y={y}, z={z}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer (runs at 10 Hz)."""

        # Continuously publish the offboard "heartbeat"
        self.publish_offboard_control_heartbeat_signal()

        # For the first ~1 second, just publish the heartbeat before switching modes
        if self.offboard_setpoint_counter < 10:
            self.offboard_setpoint_counter += 1
            return
        elif self.offboard_setpoint_counter == 10:
            # Engage Offboard and Arm once
            self.engage_offboard_mode()
            self.arm()
            self.state = 1  # start taking off
            self.offboard_setpoint_counter += 1

        # Check that we are in OFFBOARD mode
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # State machine for mission
            if self.state == 1:
                # 1) TAKE OFF to 25 ft (~ -7.62 m in NED)
                self.publish_position_setpoint(0.0, 0.0, self.altitude_m)

                # If we have reached or are below the target altitude (z <= ~ -7.5),
                # switch state to 2 (move forward).
                current_z = self.vehicle_local_position.z
                if current_z <= (self.altitude_m + 0.2):
                    self.get_logger().info("Reached takeoff altitude, transitioning to forward flight.")
                    self.state = 2

            elif self.state == 2:
                # 2) MOVE FORWARD 5 ft (~1.52 m) along +y if yaw=90 deg
                # Keep altitude at the same level
                x = 0.0
                y = self.forward_dist_m
                z = self.altitude_m
                self.publish_position_setpoint(x, y, z)

                current_x = self.vehicle_local_position.x
                current_y = self.vehicle_local_position.y
                # Check if we are close enough to the forward target
                # (for example within +/- 0.2 m)
                if (abs(current_y - y) < 0.2) and (abs(current_x - x) < 0.2):
                    self.get_logger().info("Reached forward setpoint, transitioning to land.")
                    self.state = 3

            elif self.state == 3:
                # 3) LAND
                self.land()
                # Optionally disarm after landing or exit
                # self.disarm()  # not strictly necessary if land command disarms automatically
                # exit(0)  # If you want to stop the script

        else:
            self.get_logger().warn("Not in OFFBOARD mode yet; waiting.")


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
