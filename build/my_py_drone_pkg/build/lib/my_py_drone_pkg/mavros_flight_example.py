#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

# Import service types from mavros_msgs
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
# Import message types for offboard setpoints
from geometry_msgs.msg import PoseStamped

class MavrosFlightExample(Node):
    def __init__(self):
        super().__init__('mavros_flight_example')
        self.declare_parameter("takeoff_alt", 6.1)  # 20 ft in meters
        self.declare_parameter("forward_distance", 2.44)  # 8 ft in meters

        self.takeoff_alt = self.get_parameter("takeoff_alt").value
        self.forward_distance = self.get_parameter("forward_distance").value

        # Create clients for the services
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # Wait for the services to be available
        self.get_logger().info("Waiting for MAVROS services...")
        self.arm_client.wait_for_service()
        self.set_mode_client.wait_for_service()
        self.takeoff_client.wait_for_service()
        self.land_client.wait_for_service()
        self.get_logger().info("All services available.")

        # Create a publisher for offboard position setpoints
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        # Timer to publish setpoints (required for offboard mode)
        self.timer = self.create_timer(0.1, self.publish_setpoint)
        self.current_setpoint = PoseStamped()
        # Initialize setpoint at the current (assumed) home position (x=0,y=0) and altitude 0.
        self.current_setpoint.pose.position.x = 0.0
        self.current_setpoint.pose.position.y = 0.0
        self.current_setpoint.pose.position.z = self.takeoff_alt  # target altitude for takeoff

    def publish_setpoint(self):
        # Publish the current setpoint continuously (at 10Hz) to maintain OFFBOARD mode.
        self.current_setpoint.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_pub.publish(self.current_setpoint)

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info("Drone armed successfully.")
            return True
        else:
            self.get_logger().error("Failed to arm the drone.")
            return False

    def set_offboard_mode(self):
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().mode_sent:
            self.get_logger().info("OFFBOARD mode set.")
            return True
        else:
            self.get_logger().error("Failed to set OFFBOARD mode.")
            return False

    def takeoff(self):
        req = CommandTOL.Request()
        req.altitude = self.takeoff_alt
        req.latitude = 0.0  # not used in OFFBOARD mode
        req.longitude = 0.0  # not used in OFFBOARD mode
        req.min_pitch = 0.0
        req.yaw = 0.0
        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"Takeoff to {self.takeoff_alt} m initiated.")
            return True
        else:
            self.get_logger().error("Failed to initiate takeoff.")
            return False

    def move_forward(self):
        # Update the setpoint to move forward along the x-axis by forward_distance.
        self.current_setpoint.pose.position.x = self.forward_distance
        self.get_logger().info(f"Moving forward {self.forward_distance} m.")
        # Give some time for the drone to reach the new position.
        time.sleep(5)  # Adjust delay as necessary

    def land(self):
        req = CommandTOL.Request()
        # For landing, set altitude to zero.
        req.altitude = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        future = self.land_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info("Landing initiated.")
            return True
        else:
            self.get_logger().error("Failed to initiate landing.")
            return False

def main(args=None):
    rclpy.init(args=args)
    flight_node = MavrosFlightExample()

    # Give some time for setpoint publishing before switching to OFFBOARD mode.
    flight_node.get_logger().info("Publishing initial setpoints...")
    time.sleep(2)

    # Set offboard mode
    if not flight_node.set_offboard_mode():
        rclpy.shutdown()
        return

    # Arm the drone
    if not flight_node.arm():
        rclpy.shutdown()
        return

    # Initiate takeoff
    if not flight_node.takeoff():
        rclpy.shutdown()
        return

    # Wait for the drone to reach takeoff altitude.
    flight_node.get_logger().info("Waiting for takeoff to complete...")
    time.sleep(5)  # Adjust delay based on your system's response

    # Move forward by updating the setpoint
    flight_node.move_forward()

    # Land the drone
    flight_node.land()

    # Give some time for landing to complete before shutting down.
    flight_node.get_logger().info("Waiting for landing to complete...")
    time.sleep(5)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
