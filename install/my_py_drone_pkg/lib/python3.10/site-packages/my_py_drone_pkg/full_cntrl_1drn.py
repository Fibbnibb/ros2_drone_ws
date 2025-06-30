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
import math

class XYDroneController(Node):
    def __init__(self, namespace="drone1"):
        super().__init__(f"xy_control_{namespace}")
        
        # Store namespace
        self.namespace = namespace
        
        # Create callback groups
        self.timer_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()
        
        # Mission state
        self.mission_in_progress = False
        self.current_mission_step = 0
        self.mission_completed = False

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriptions
        self.state_sub = self.create_subscription(
            State, f'/{namespace}/state', self.state_callback, 10)
        
        self.pose_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.pose_callback, qos_profile)
        
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
        
        # Position setpoint publisher
        self.position_pub = self.create_publisher(
            PoseStamped, f'/{namespace}/setpoint_position/local', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for arming service for {namespace}...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for set mode service for {namespace}...")
            
        # Initialize state and pose
        self.current_state = State()
        self.current_pose = PoseStamped()
        
        # Initialize setpoint
        self.setpoint = PoseStamped()
        self.setpoint.header.frame_id = "map"
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = 2.0  # Default takeoff height
        self.setpoint.pose.orientation.w = 1.0  # No rotation
        
        # Create timers with callback groups
        self.setpoint_timer = self.create_timer(
            0.1, 
            self.publish_setpoint, 
            callback_group=self.timer_group
        )
        
        self.mission_timer = self.create_timer(
            0.5,  # Check mission progress every 0.5 seconds
            self.mission_step,
            callback_group=self.timer_group
        )
        
        # Target position and tolerance
        self.target_position = {
            'x': 0.0,
            'y': 0.0,
            'z': 2.0
        }
        self.position_tolerance = 0.2
        self.position_timeout = 30.0  # Seconds
        self.position_start_time = 0.0
        
        # Mission waypoints for square pattern
        self.square_waypoints = []
        self.current_waypoint_index = 0
        
        self.get_logger().info(f"XY controller initialized for {namespace}")
    
    def pose_callback(self, msg):
        """Store current pose"""
        self.current_pose = msg
    
    def state_callback(self, msg):
        """Store current state"""
        self.current_state = msg
    
    def publish_setpoint(self):
        """Timer callback to publish position setpoint"""
        self.setpoint.header.stamp = self.get_clock().now().to_msg()
        self.position_pub.publish(self.setpoint)
    
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
                self.get_logger().info("Vehicle armed successfully")
                # After arming, proceed to next step in mission if needed
                if self.mission_in_progress and self.current_mission_step == 1:  # After OFFBOARD mode
                    self.current_mission_step = 2  # Proceed to takeoff
            else:
                self.get_logger().warn("Arming failed, retrying...")
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
                self.get_logger().info(f"Mode {requested_mode} set successfully")
                
                # After setting OFFBOARD mode, proceed to arming
                if self.mission_in_progress and requested_mode == "OFFBOARD" and self.current_mission_step == 0:
                    self.current_mission_step = 1  # Ready to arm
                
                # If switching to AUTO.LAND mode
                if requested_mode == "AUTO.LAND":
                    self.get_logger().info("Landing initiated")
            else:
                self.get_logger().warn(f"Failed to set mode {requested_mode}, retrying...")
                # Retry setting mode
                self.set_mode_async(requested_mode)
        except Exception as e:
            self.get_logger().error(f"Set mode service call failed: {e}")
    
    def is_at_position(self, x, y, z, tolerance):
        """Check if drone is at the specified position within tolerance"""
        if not hasattr(self, 'current_pose') or self.current_pose is None:
            return False
            
        dx = abs(self.current_pose.pose.position.x - x)
        dy = abs(self.current_pose.pose.position.y - y)
        dz = abs(self.current_pose.pose.position.z - z)
        
        return dx < tolerance and dy < tolerance and dz < tolerance
    
    def update_target_position(self, x, y, z=None):
        """Update the target position"""
        # If z is not specified, maintain current height setpoint
        if z is None:
            z = self.setpoint.pose.position.z
            
        self.target_position['x'] = x
        self.target_position['y'] = y
        self.target_position['z'] = z
        
        # Update setpoint
        self.setpoint.pose.position.x = x
        self.setpoint.pose.position.y = y
        self.setpoint.pose.position.z = z
        
        # Reset position tracking timer
        self.position_start_time = time.time()
        
        self.get_logger().info(f"Target position updated: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    
    def start_square_pattern(self, size=5.0, height=2.0):
        """Start a square pattern mission"""
        self.get_logger().info(f"Starting square pattern mission: size={size}m, height={height}m")
        
        # Define the square waypoints
        self.square_waypoints = [
            {'x': 0.0, 'y': 0.0, 'z': height, 'wait': 3.0},  # Takeoff position
            {'x': size, 'y': 0.0, 'z': height, 'wait': 2.0},  # First corner
            {'x': size, 'y': size, 'z': height, 'wait': 2.0},  # Second corner
            {'x': 0.0, 'y': size, 'z': height, 'wait': 2.0},  # Third corner
            {'x': 0.0, 'y': 0.0, 'z': height, 'wait': 2.0}   # Back to start
        ]
        
        self.current_waypoint_index = 0
        self.mission_in_progress = True
        self.mission_completed = False
        self.current_mission_step = 0  # Start with setting OFFBOARD mode
        
        # Set initial target for takeoff
        self.update_target_position(0.0, 0.0, height)
        
        self.get_logger().info("Mission started")
    
    def mission_step(self):
        """Mission state machine, called periodically by timer"""
        if not self.mission_in_progress:
            return
            
        current_time = time.time()
        
        # Takeoff sequence (OFFBOARD -> ARM -> TAKEOFF)
        if self.current_mission_step == 0:
            # Step 0: Set OFFBOARD mode
            self.get_logger().info("Setting OFFBOARD mode")
            self.set_mode_async("OFFBOARD")
            
        elif self.current_mission_step == 1:
            # Step 1: Arm the vehicle
            self.get_logger().info("Arming vehicle")
            self.arm_async()
            
        elif self.current_mission_step == 2:
            # Step 2: Wait for takeoff to complete
            if self.is_at_position(
                self.target_position['x'], 
                self.target_position['y'], 
                self.target_position['z'],
                self.position_tolerance
            ):
                self.get_logger().info("Takeoff complete")
                self.current_mission_step = 3  # Move to waypoint navigation
                self.waypoint_start_time = current_time
                
            elif current_time - self.position_start_time > self.position_timeout:
                self.get_logger().warn("Takeoff timed out, continuing anyway")
                self.current_mission_step = 3  # Move to waypoint navigation
                self.waypoint_start_time = current_time
                
        elif self.current_mission_step == 3:
            # Step 3: Navigate through waypoints
            if self.current_waypoint_index < len(self.square_waypoints):
                waypoint = self.square_waypoints[self.current_waypoint_index]
                
                # Check if we're at the current waypoint
                if self.is_at_position(
                    waypoint['x'], waypoint['y'], waypoint['z'], 
                    self.position_tolerance
                ):
                    # If we just arrived, start the wait timer
                    if not hasattr(self, 'waypoint_start_time'):
                        self.waypoint_start_time = current_time
                        self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}")
                    
                    # Wait at waypoint for specified time
                    elif current_time - self.waypoint_start_time >= waypoint['wait']:
                        # Move to next waypoint
                        self.current_waypoint_index += 1
                        self.get_logger().info(f"Moving to waypoint {self.current_waypoint_index}")
                        
                        if self.current_waypoint_index < len(self.square_waypoints):
                            next_waypoint = self.square_waypoints[self.current_waypoint_index]
                            self.update_target_position(
                                next_waypoint['x'], next_waypoint['y'], next_waypoint['z']
                            )
                        else:
                            # All waypoints visited, proceed to landing
                            self.current_mission_step = 4
                            self.get_logger().info("All waypoints visited, initiating landing")
                
                # Check for timeout moving to waypoint
                elif current_time - self.position_start_time > self.position_timeout:
                    self.get_logger().warn(f"Timed out reaching waypoint {self.current_waypoint_index}, moving to next")
                    self.current_waypoint_index += 1
                    
                    if self.current_waypoint_index < len(self.square_waypoints):
                        next_waypoint = self.square_waypoints[self.current_waypoint_index]
                        self.update_target_position(
                            next_waypoint['x'], next_waypoint['y'], next_waypoint['z']
                        )
                    else:
                        # All waypoints visited, proceed to landing
                        self.current_mission_step = 4
                        self.get_logger().info("All waypoints visited, initiating landing")
            
        elif self.current_mission_step == 4:
            # Step 4: Landing
            self.get_logger().info("Setting AUTO.LAND mode")
            self.set_mode_async("AUTO.LAND")
            self.current_mission_step = 5
            
        elif self.current_mission_step == 5:
            # Step 5: Wait for landing to complete
            if not self.current_state.armed:
                self.get_logger().info("Landing complete, mission finished")
                self.mission_completed = True
                self.mission_in_progress = False
            elif current_time - self.position_start_time > 30.0:  # 30 seconds timeout for landing
                self.get_logger().warn("Landing timed out")
                self.mission_completed = True
                self.mission_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    
    # Create drone controller
    controller = XYDroneController("drone1")
    
    # Create executor with multiple threads
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    
    # Mission control timer - create a timer in the main thread
    # that periodically checks mission status without blocking execution
    mission_complete = False
    
    try:
        # Wait for everything to initialize
        print("Initializing drone controller...")
        for _ in range(20):  # 2 seconds of initialization
            executor.spin_once(timeout_sec=0.1)
        
        # Start the square pattern mission
        print("Starting square pattern mission")
        controller.start_square_pattern(5.0, 2.0)
        
        # Main execution loop - keep spinning the executor until mission complete
        print("Mission in progress...")
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            
            # Check if mission is complete
            if controller.mission_completed:
                print("Mission completed successfully!")
                break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        # Set AUTO.LAND mode if interrupted
        controller.set_mode_async("AUTO.LAND")
        # Process a few more callbacks to ensure the land command is sent
        for _ in range(10):
            executor.spin_once(timeout_sec=0.1)
        
    finally:
        # Shutdown
        rclpy.shutdown()
        
    print("Controller terminated")

if __name__ == '__main__':
    main()