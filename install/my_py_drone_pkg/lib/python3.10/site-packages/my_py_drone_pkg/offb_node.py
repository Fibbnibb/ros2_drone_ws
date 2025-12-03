#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

class OffboardNode(Node):
    def __init__(self):
        super().__init__("offb_node_py")

        # Current MAVROS state (initialized to default)
        self.current_state = State()

        # Subscriber to MAVROS state topic
        self.create_subscription(
            State,
            "mavros/state",
            self.state_cb,
            10
        )

        # Publisher for local position setpoints
        self.local_pos_pub = self.create_publisher(
            PoseStamped,
            "mavros/setpoint_position/local",
            10
        )

        # Create service clients for arming and mode switching
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /mavros/cmd/arming service...")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /mavros/set_mode service...")

        # Set the publishing rate (20 Hz)
        self.rate_hz = 20.0
        self.timer_period = 1.0 / self.rate_hz

        # Record the last request time
        self.last_req = self.get_clock().now()

        # Create and initialize the setpoint message
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0.0
        self.pose.pose.position.y = 0.0
        self.pose.pose.position.z = 2.0

        # Send a few setpoints before switching modes (similar to the 100-loop in ROS 1)
        self.get_logger().info("Sending initial setpoints...")
        for _ in range(100):
            self.pose.header.stamp = self.get_clock().now().to_msg()
            self.local_pos_pub.publish(self.pose)
            rclpy.spin_once(self, timeout_sec=0.01)

        # Create a timer to run the main loop at 20 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def state_cb(self, msg: State):
        """Callback to update current MAVROS state."""
        self.current_state = msg

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_req).nanoseconds / 1e9  # Convert nanoseconds to seconds

        # If the mode is not OFFBOARD, attempt to switch every 5 seconds
        if self.current_state.mode != "OFFBOARD" and elapsed > 5.0:
            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"
            future = self.set_mode_client.call_async(req)
            future.add_done_callback(self.mode_response_callback)
            self.last_req = now

        # Otherwise, if not armed, attempt to arm every 5 seconds
        elif (not self.current_state.armed) and elapsed > 5.0:
            req = CommandBool.Request()
            req.value = True
            future = self.arm_client.call_async(req)
            future.add_done_callback(self.arm_response_callback)
            self.last_req = now

        # Always publish the current setpoint to maintain OFFBOARD mode
        self.pose.header.stamp = now.to_msg()
        self.local_pos_pub.publish(self.pose)

    def mode_response_callback(self, future):
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info("OFFBOARD enabled")
        except Exception as e:
            self.get_logger().error(f"Failed to set mode: {e}")

    def arm_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Vehicle armed")
        except Exception as e:
            self.get_logger().error(f"Failed to arm: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OffboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
