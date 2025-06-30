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
        
        self.drone_hieght_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.drone_hieght_callback, 10)
        
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
    
    def drone_hieght_callback(self, msg):
        self.current_pose = msg

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

    def takeoff(self, height):
        
        #Takeoff function that gets drone to specified height
        self.get_logger().info("Initiating takeoff sequence...")
        # Set the takeoff height
        self.setpoint.pose.position.z = height
        
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
        
        # Keep publishing setpoints for takeoff for 5 seconds
        self.get_logger().info(f"Taking off to {height} meters...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            self.publish_setpoint()
            time.sleep(0.1)
            
        self.get_logger().info("Takeoff complete")
    
    def move_forward(self, distance):
        #Move forward function to move the drone forward
        self.get_logger().info(f"Moving forward {distance} meters")
        self.setpoint.pose.position.x = distance

    def land(self):
        #Land function to safely land the drone
        self.get_logger().info("Initiating landing sequence...")
        
        # Switch to land mode
        self.set_mode("AUTO.LAND")
        
        # Wait for landing to complete
        start_time = time.time()
        while time.time() - start_time < 10.0:  # Wait up to 10 seconds for landing
            self.publish_setpoint()
            time.sleep(0.1)
            
        self.get_logger().info("Landing complete")
    
    def takeoff_move_and_land(self):
        # Run the takeoff and land sequence
        self.takeoff(2.0)
        start_time = time.time()
        while time.time() - start_time < 10.0:  # Wait up to 10 seconds for landing
            self.publish_setpoint()
            time.sleep(0.1)
        self.move_forward(5.0)
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publish_setpoint()
            time.sleep(0.1)
        #time.sleep(5.0)
        self.land()

def main(args=None):
    rclpy.init(args=args)
    drone = Drone()
    # Run the takeoff and land sequence
    drone.takeoff_move_and_land()
    # Keep the node running
    #rclpy.spin(drone)
    drone.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()