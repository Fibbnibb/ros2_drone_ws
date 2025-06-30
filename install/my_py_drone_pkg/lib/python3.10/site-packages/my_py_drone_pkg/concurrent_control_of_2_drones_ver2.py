#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import threading

class Drone(Node):
    def __init__(self, namespace):
        super().__init__(f"drone_node_{namespace}")
        
        self.namespace = namespace
        
        # Configure QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Update topic paths to match actual ROS2 structure
        self.state_sub = self.create_subscription(
            State, f'/{namespace}/state', self.state_callback, 10)
        
        self.drone_height_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.drone_height_callback, qos_profile)
        
        self.arm_client = self.create_client(CommandBool, f'/{namespace}/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, f'/{namespace}/set_mode')
        
        self.position_pub = self.create_publisher(
            PoseStamped, f'/{namespace}/setpoint_position/local', 10)

        # Wait for services with correct paths
        self.get_logger().info(f"Waiting for services for {namespace}...")
        self.arm_client.wait_for_service()
        self.set_mode_client.wait_for_service()
        self.get_logger().info(f"All services available for {namespace}")
            
        # Initialize state and pose
        self.current_state = State()
        self.current_pose = PoseStamped()
        
        # Create and initialize setpoint message
        self.setpoint = PoseStamped()
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = 2.0
        self.setpoint.pose.orientation.w = 1.0
        
        # Create timer for setpoint publishing
        self.timer = self.create_timer(0.1, self.publish_setpoint)
    

    def wait_for_connection(self):
        """Wait for MAVROS connection"""
        timeout = 0
        while not self.current_state.connected and timeout < 30:
            self.get_logger().info(f"Waiting for {self.namespace} to connect...")
            time.sleep(1)
            timeout += 1
            rclpy.spin_once(self)
        
        if self.current_state.connected:
            self.get_logger().info(f"{self.namespace} connected!")
            return True
        else:
            self.get_logger().error(f"{self.namespace} failed to connect!")
            return False

    # ... [Other methods remain the same] ...
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
        
        #Takeoff function that gets drone to specified height
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
        self.get_logger().info("Vehicle armed")
        
        # Keep publishing setpoints for takeoff for 5 seconds
        self.get_logger().info(f"{self.namespace} Taking off to {height} meters...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            self.publish_setpoint()
            time.sleep(0.1)
            
        self.get_logger().info(f"{self.namespace} takeoff complete")
    
    def move_forward(self, distance):
        #Move forward function to move the drone forward
        self.get_logger().info(f"{self.namespace} Moving forward {distance} meters")
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

    def takeoff_move_and_land(self, forward_distance):
        self.get_logger().info(f"{self.namespace}: Starting mission...")
        # Run the takeoff and land sequence
        self.takeoff(2.0)
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publish_setpoint()
            time.sleep(0.1)
        
        #move the drone forward
        self.move_forward(forward_distance)
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publish_setpoint()
            time.sleep(0.1)
        #land the drone
        self.land()

def main(args=None):
    rclpy.init(args=args)
    
    # Create two drone instances
    drone1 = Drone("drone1")
    drone2 = Drone("drone2")
    
    # Wait for both drones to be ready
    time.sleep(2)
    
    if not (drone1.wait_for_connection() and drone2.wait_for_connection()):
        print("Failed to connect to one or both drones!")
        return

    # Create threads for each drone's mission
    thread1 = threading.Thread(target=drone1.takeoff_move_and_land, args=(5.0,))
    thread2 = threading.Thread(target=drone2.takeoff_move_and_land, args=(10.0,))
    
    try:
        # Start both missions concurrently
        thread1.start()
        time.sleep(1)  # Small delay between drone starts
        thread2.start()
        
        # Keep the ROS nodes running
        while thread1.is_alive() or thread2.is_alive():
            rclpy.spin_once(drone1)
            rclpy.spin_once(drone2)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        drone1.land()
        drone2.land()
    finally:
        drone1.destroy_node()
        drone2.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()