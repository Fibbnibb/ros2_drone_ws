#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode with a simple state machine."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,  # <-- changed
            history=HistoryPolicy.KEEP_LAST,
            depth=1000  # or some larger depth
        )

        qos_profile_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000
        )

        qos_profile_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000
        )

        # --- Publishers ---
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # --- Subscribers ---
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

        # --- Internal state ---
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        self.takeoff_height = -5.0  # 5 meters above ground in NED (Z is negative up)
        self.forward_distance = 5.0 # Move +5m in X

        self.offboard_setpoint_counter = 0
        self.flight_state = 0  # 0: Init/Takeoff, 1: Move forward, 2: Wait 2s, 3: Land
        self.forward_counter = 0  # Used to count time steps for forward flight
        self.forward_time_limit = 20  # 2 seconds / 0.1s timer callback = 20 cycles

        # Create a timer to publish control commands every 0.1 s
        self.timer = self.create_timer(0.1, self.timer_callback)

    # --- Callbacks ---
    def vehicle_local_position_callback(self, msg):
        """Callback for vehicle_local_position."""
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        """Callback for vehicle_status."""
        self.vehicle_status = msg

    # --- Command methods ---
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        # param1 = 1 => custom mode
        # param2 = 6 => offboard
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    # --- Publishing methods ---
    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control heartbeat (OffboardControlMode)."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float = 1.57079):
        """
        Publish a position setpoint.
        
        :param x: position in X (forward in NED frame)
        :param y: position in Y (right in NED frame)
        :param z: position in Z (down in NED frame; negative is 'up')
        :param yaw: yaw in radians (default ~90 deg)
        """
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoint: x={x}, y={y}, z={z}")

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

    # --- Main control logic in timer callback ---
    def timer_callback(self) -> None:
        """Periodic callback to handle offboard control logic."""
        # Always publish offboard heartbeat
        self.publish_offboard_control_heartbeat_signal()

        # After a few ticks, switch to offboard and arm
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        # Simple state machine
        #
        # flight_state = 0 => Initial, ascend to takeoff_height
        # flight_state = 1 => Move forward 5 meters
        # flight_state = 2 => After 2 seconds forward, land
        # flight_state = 3 => End (no more commands)
        
        # Make sure the vehicle is actually in OFFBOARD navigation
        # (NAVIGATION_STATE_OFFBOARD = 14 typically, but using the status from the message)
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:

            # --- State 0: Take off ---
            if self.flight_state == 0:
                # Command to maintain position at (0,0,takeoff_height)
                self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
                
                # Check if we've reached near the takeoff altitude
                if self.vehicle_local_position.z <= (self.takeoff_height + 0.2):
                    self.get_logger().info("Reached takeoff altitude, transitioning to forward flight.")
                    self.flight_state = 1

            # --- State 1: Move forward 5 meters ---
            elif self.flight_state == 1:
                # Keep the same altitude, but move +5 in X
                self.publish_position_setpoint(self.forward_distance, 0.0, self.takeoff_height)
                
                # Count how many ticks we've been in forward flight
                self.forward_counter += 1

                # Once we've been here for 2 seconds (20 ticks at 0.1s each)
                if self.forward_counter >= self.forward_time_limit:
                    self.get_logger().info("Completed 2s forward flight, preparing to land.")
                    self.flight_state = 2

            # --- State 2: Initiate landing ---
            elif self.flight_state == 2:
                self.land()
                self.flight_state = 3

        # Count up so we eventually switch to offboard mode and arm
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


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