#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time

class Drone(Node):
    def __init__(self, namespace):
        super().__init__(f"drone_node_{namespace}")
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.mission_completed = False
        
        # Create callback groups - one for timer, one for services
        self.timer_group = ReentrantCallbackGroup() # ReentrantCallbackGroup act concurrently or in parallel best for timers
        self.service_group = MutuallyExclusiveCallbackGroup() # MutuallyExclusiveCallbackGroup act sequentially best for services

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
        
        # Service clients with callback group
        self.arm_client = self.create_client(
            CommandBool, 
            f'/{namespace}/cmd/arming',
            callback_group=self.service_group
        )
        
        self.set_mode_client = self.create_client(
            SetMode, 
            f'/{namespace}/set_mode',
            callback_group=self.service_group
        )
        
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
        self.current_pose = PoseStamped()
        
        self.setpoint = PoseStamped()
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = 2.0
        self.setpoint.pose.orientation.w = 1.0
        
        # Store home position
        self.home_position = PoseStamped()
        self.home_position.pose.position.x = 0.0
        self.home_position.pose.position.y = 0.0
        self.home_position.pose.position.z = 2.0
        self.home_position.pose.orientation.w = 1.0
        
        # Mission state variables
        self.mission_state = "IDLE"
        self.target_distance = 0.0
        self.state_start_time = 0.0
        self.last_setpoint_time = self.get_clock().now()
        
        # Create timers with their callback group
        self.setpoint_timer = self.create_timer(
            0.1, 
            self.publish_setpoint, 
            callback_group=self.timer_group
        )
        
        self.mission_timer = self.create_timer(
            0.1, 
            self.mission_step, 
            callback_group=self.timer_group
        )
        
        self.get_logger().info(f"{namespace} initialized")

    def drone_height_callback(self, msg):
        self.current_pose = msg

    def state_callback(self, msg):
        self.current_state = msg

    def publish_setpoint(self):
        """Timer callback to publish setpoint"""
        now = self.get_clock().now()
        self.setpoint.header.stamp = now.to_msg()
        self.position_pub.publish(self.setpoint)
        self.last_setpoint_time = now

    def arm_async(self):
        """Non-blocking arm request"""
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        future.add_done_callback(self.arm_done_callback)
    
    def arm_done_callback(self, future):
        """Callback for arm service response"""
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"{self.namespace} Armed successfully")
                if self.mission_state == "ARMING":
                    self.mission_state = "SET_OFFBOARD"
                    self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.get_logger().warn(f"{self.namespace} Arming failed")
                # Retry arming
                self.arm_async()
        except Exception as e:
            self.get_logger().error(f"Arm service call failed: {e}")
    
    def set_mode_async(self, mode):
        """Non-blocking mode change request"""
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(lambda f: self.mode_done_callback(f, mode))
    
    def mode_done_callback(self, future, requested_mode):
        """Callback for set_mode service response"""
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info(f"{self.namespace} Mode {requested_mode} set successfully")
                
                if self.mission_state == "SET_OFFBOARD" and requested_mode == "OFFBOARD":
                    self.mission_state = "TAKEOFF"
                    self.state_start_time = self.get_clock().now().nanoseconds / 1e9
                
                elif self.mission_state == "LANDING" and requested_mode == "AUTO.LAND":
                    # Let landing progress through timer
                    pass
            else:
                self.get_logger().warn(f"{self.namespace} Failed to set mode {requested_mode}")
                # Retry mode setting
                self.set_mode_async(requested_mode)
        except Exception as e:
            self.get_logger().error(f"Set mode service call failed: {e}")
    
    def start_mission(self, forward_distance):
        """Start the mission state machine"""
        self.get_logger().info(f"{self.namespace} Starting mission...")
        self.mission_state = "INIT"
        self.target_distance = forward_distance
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
    
    def return_to_home(self):
        """Initiate return to home procedure"""
        self.get_logger().info(f"{self.namespace} Returning to home position...")
        self.mission_state = "RETURN_HOME"
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Reset setpoint to current position first to avoid sudden jumps
        self.setpoint.pose.position.x = self.current_pose.pose.position.x
        self.setpoint.pose.position.y = self.current_pose.pose.position.y
        self.setpoint.pose.position.z = self.current_pose.pose.position.z
    
    def mission_step(self):
        """Mission state machine timer callback"""
        if self.mission_state == "IDLE":
            # Do nothing
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.state_start_time
            
        if self.mission_state == "INIT":
            # Send setpoints for a while before starting
            if elapsed > 2.0:  # 2 seconds of setpoints
                self.mission_state = "ARMING"
                self.get_logger().info(f"{self.namespace} Initiating arming sequence")
                self.arm_async()
        
        elif self.mission_state == "ARMING":
            # Wait for arm_done_callback to advance state
            pass
            
        elif self.mission_state == "SET_OFFBOARD":
            # Request OFFBOARD mode
            self.set_mode_async("OFFBOARD")
            
        elif self.mission_state == "TAKEOFF":
            # In takeoff state, wait for drone to reach altitude
            self.setpoint.pose.position.z = 2.0
            
            # After 5 seconds, or when altitude is reached, transition to forward movement
            if elapsed > 5.0:
                self.mission_state = "MOVE_FORWARD"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, moving forward")
        
        elif self.mission_state == "MOVE_FORWARD":
            # Move forward
            self.setpoint.pose.position.x = self.target_distance
            
            # After 10 seconds, start return to home
            if elapsed > 10.0:
                self.mission_state = "RETURN_HOME"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Forward movement complete, returning home")
                
        elif self.mission_state == "RETURN_HOME":
            # Gradually move back to home position (X=0, Y=0)
            # Calculate how far we've gone in return journey as a percentage (0-1)
            return_progress = min(elapsed / 10.0, 1.0)  # Complete return in 10 seconds
            
            # Linear interpolation between current forward position and home
            target_x = self.target_distance * (1.0 - return_progress)
            self.setpoint.pose.position.x = target_x
            
            # When we've completed the return or 10 seconds have passed
            if elapsed > 10.0:
                self.mission_state = "LANDING"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            # Wait for landing to complete
            if elapsed > 10.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")

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
        # Start missions
        drone1.start_mission(5.0)   # First drone moves 5 meters
        drone2.start_mission(10.0)  # Second drone moves 10 meters
        
        # Spin to allow missions to progress
        while not (drone1.mission_completed and drone2.mission_completed):
            executor.spin_once()
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
    except KeyboardInterrupt:
        # Return to home for both drones if interrupted
        drone1.return_to_home()
        drone2.return_to_home()
        # Give some time for return to begin
        time.sleep(2.0)
        
        # Continue processing until we decide to exit
        end_time = time.time() + 5.0  # Process for up to 5 more seconds
        while time.time() < end_time:
            executor.spin_once()
            time.sleep(0.01)
            
        # Finally land the drones
        drone1.set_mode_async("AUTO.LAND")
        drone2.set_mode_async("AUTO.LAND")
        time.sleep(2.0)
        
    finally:
        # Cleanup
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()