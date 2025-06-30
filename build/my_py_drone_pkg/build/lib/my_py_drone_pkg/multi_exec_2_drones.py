#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import time

class Drone(Node):
    def __init__(self, namespace):
        super().__init__(f"drone_node_{namespace}")
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.mission_completed = False
        self.mission_thread = None

        # Create callback group
        self.callback_group = ReentrantCallbackGroup()

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriptions
        self.state_sub = self.create_subscription(
            State, f'/{namespace}/state', self.state_callback, 10)
        
        self.drone_height_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.drone_height_callback, qos_profile)
        
        # Service clients
        self.arm_client = self.create_client(CommandBool, f'/{namespace}/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, f'/{namespace}/set_mode')
        
        # Setpoint publisher
        self.position_pub = self.create_publisher(
            PoseStamped, f'/{namespace}/setpoint_position/local', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for arming service for {namespace}...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for set mode service for {namespace}...")
            
        # Initialize state and setpoint
        self.current_state = State()
        self.setpoint = PoseStamped()
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = 2.0
        self.setpoint.pose.orientation.w = 1.0
        
        # Setpoint publishing timer
        self.timer = self.create_timer(0.1, self.publish_setpoint, callback_group=self.callback_group)

    def drone_height_callback(self, msg):
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
        """Takeoff function that gets drone to specified height"""
        self.get_logger().info(f"{self.namespace} Initiating takeoff sequence...")
        
        # Set the takeoff height
        self.setpoint.pose.position.z = height

        # Send some setpoints before starting
        for _ in range(20):  # Wait 2 seconds
            self.publish_setpoint()
            time.sleep(0.1)
            
        # Switch to offboard mode
        self.set_mode("OFFBOARD")
        self.get_logger().info(f"{self.namespace} OFFBOARD mode requested")
        time.sleep(1)
        
        # Arm the vehicle
        self.arm()
        self.get_logger().info(f"{self.namespace} Vehicle armed")
        
        # Keep publishing setpoints for takeoff for 5 seconds
        self.get_logger().info(f"{self.namespace} Taking off to {height} meters...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            self.publish_setpoint()
            time.sleep(0.1)
            
        self.get_logger().info(f"{self.namespace} takeoff complete")
    
    def move_forward(self, distance):
        """Move forward function to move the drone forward"""
        self.get_logger().info(f"{self.namespace} Moving forward {distance} meters")
        self.setpoint.pose.position.x = distance
        
        # Maintain position for some time
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publish_setpoint()
            time.sleep(0.1)
    
    def land(self):
        """Land function to safely land the drone"""
        self.get_logger().info(f"{self.namespace} Initiating landing sequence...")
        
        # Switch to land mode
        self.set_mode("AUTO.LAND")
        
        # Wait for landing to complete
        start_time = time.time()
        while time.time() - start_time < 10.0:  # Wait up to 10 seconds for landing
            self.publish_setpoint()
            time.sleep(0.1)
            
        self.mission_completed = True
        self.get_logger().info(f"{self.namespace} Landing complete")

    def start_mission(self, forward_distance):
        """Start mission in a separate thread"""
        def mission_thread_func():
            """Mission sequence in a separate thread"""
            self.get_logger().info(f"{self.namespace}: Starting mission...")
            
            # Run the takeoff and land sequence
            self.takeoff(2.0)
            
            # Move forward
            self.move_forward(forward_distance)
            
            # Land the drone
            self.land()
        
        # Create and start the mission thread
        self.mission_thread = threading.Thread(target=mission_thread_func)
        self.mission_thread.start()

def main(args=None):
    rclpy.init(args=args)
    
    # Create drone instances
    drone1 = Drone("drone1")
    drone2 = Drone("drone2")
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(drone1)
    executor.add_node(drone2)
    
    try:
        # Start missions concurrently
        drone1.start_mission(5.0)   # First drone moves 5 meters
        drone2.start_mission(10.0)  # Second drone moves 10 meters
        
        # Spin to allow missions to progress
        while not (drone1.mission_completed and drone2.mission_completed):
            executor.spin_once()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        # Safe landing for both drones if interrupted
        drone1.land()
        drone2.land()
    finally:
        # Cleanup
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()