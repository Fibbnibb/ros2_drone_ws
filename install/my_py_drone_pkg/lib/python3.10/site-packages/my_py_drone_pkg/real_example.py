#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
import time

class Drone(Node):
    def __init__(self):
        super().__init__("drone_node")
        
        # Subscribe to drone state
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, 10)
        
        # Create service clients
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        # Create setpoint publisher
        self.position_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arming service...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for set mode service...")
            
        # Initialize state
        self.current_state = State()
        
        # Create and initialize setpoint message
        self.setpoint = PoseStamped()
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = 2.0  # Target height
        self.setpoint.pose.orientation.w = 1.0  # Simple orientation
        
        # Create timer for setpoint publishing
        self.timer = self.create_timer(0.1, self.publish_setpoint)  # 10Hz

    def state_callback(self, msg):
        self.current_state = msg
        
    def publish_setpoint(self):
        self.setpoint.header.stamp = self.get_clock().now().to_msg()
        self.position_pub.publish(self.setpoint)
        
    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent
        
    def arm(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success
        
    def takeoff_and_land(self):
        # Send some setpoints before starting
        for _ in range(20):  # Wait 2 seconds
            self.publish_setpoint()
            time.sleep(0.1)
            
        # Switch to offboard mode
        self.set_mode("OFFBOARD")
        self.get_logger().info("OFFBOARD mode requested")
        time.sleep(1)
        
        # Arm the vehicle
        self.arm()
        self.get_logger().info("Vehicle armed")
        
        # Keep publishing setpoints for 10 seconds
        self.get_logger().info("Taking off...")
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publish_setpoint()
            time.sleep(0.1)
            
        # Switch to land mode
        self.set_mode("AUTO.LAND")
        self.get_logger().info("Landing...")

def main(args=None):
    rclpy.init(args=args)
    drone = Drone()
    
    # Run the takeoff and land sequence
    drone.takeoff_and_land()
    
    # Keep the node running
    rclpy.spin(drone)
    
    # Clean up
    drone.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()