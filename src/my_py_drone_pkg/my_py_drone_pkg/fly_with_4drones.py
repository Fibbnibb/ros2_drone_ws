#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
import numpy as np

class DroneState:
    """Class to store and handle drone position, orientation, and velocity data"""
    def __init__(self, drone_id):
        self.drone_id = drone_id
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w] quaternion
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.speed = 0.0
        self.last_update_time = 0.0
        
    def update_from_pose(self, pose_msg):
        """Update position and orientation from a PoseStamped message"""
        self.position[0] = pose_msg.pose.position.x
        self.position[1] = pose_msg.pose.position.y
        self.position[2] = pose_msg.pose.position.z
        
        self.orientation[0] = pose_msg.pose.orientation.x
        self.orientation[1] = pose_msg.pose.orientation.y
        self.orientation[2] = pose_msg.pose.orientation.z
        self.orientation[3] = pose_msg.pose.orientation.w
        
    def update_from_twist(self, twist_msg):
        """Update velocity from a TwistStamped message"""
        self.velocity[0] = twist_msg.twist.linear.x
        self.velocity[1] = twist_msg.twist.linear.y
        self.velocity[2] = twist_msg.twist.linear.z
        self.speed = np.linalg.norm(self.velocity[:2])
        
    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([x, y, z, w])
    
    def set_target_orientation(self, target_roll=0.0, target_pitch=0.0, target_yaw=0.0):
        """Set target orientation using roll, pitch, yaw angles (in radians)"""
        # Apply safety limits
        max_roll = math.radians(45)   # 45 degrees max roll
        max_pitch = math.radians(30)  # 30 degrees max pitch
        
        target_roll = np.clip(target_roll, -max_roll, max_roll)
        target_pitch = np.clip(target_pitch, -max_pitch, max_pitch)
        
        # Convert to quaternion and set
        self.orientation = self.euler_to_quaternion(target_roll, target_pitch, target_yaw)
        return self.orientation
    
    def to_pose_stamped(self, stamp=None):
        """Convert to PoseStamped message"""
        pose = PoseStamped()
        if stamp is not None:
            pose.header.stamp = stamp
        pose.header.frame_id = self.drone_id
        
        pose.pose.position.x = self.position[0]
        pose.pose.position.y = self.position[1]
        pose.pose.position.z = self.position[2]
        
        pose.pose.orientation.x = self.orientation[0]
        pose.pose.orientation.y = self.orientation[1]
        pose.pose.orientation.z = self.orientation[2]
        pose.pose.orientation.w = self.orientation[3]
        
        return pose

class SimpleDroneControl(Node):
    """Simple node to control a single drone with roll, pitch, yaw control"""
    def __init__(self, namespace):
        super().__init__(f"simple_drone_{namespace}")
        
        # Basic parameters
        self.namespace = namespace
        self.mission_completed = False
        
        # Control parameters
        self.use_orientation_control = True
        self.smooth_orientation = True
        self.orientation_smoothing = 0.2  # Faster response for manual control
        
        # Demonstration parameters - THESE ARE THE VALUES YOU CAN CHANGE
        self.demo_roll = 0.0      # Target roll in degrees
        self.demo_pitch = 0.0     # Target pitch in degrees  
        self.demo_yaw = 0.0       # Target yaw in degrees
        self.demo_altitude = 2.0  # Target altitude in meters
        
        # Safety parameters
        self.position_safety_check = True
        self.max_position_jump = 5.5
        
        # Create callback groups
        self.timer_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()

        # QoS Profile for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscriptions
        self.state_sub = self.create_subscription(
            State, f'/{namespace}/state', self.state_callback, 10)
        
        self.drone_pose_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.drone_pose_callback, qos_profile)
            
        self.drone_vel_sub = self.create_subscription(
            TwistStamped, f'/{namespace}/local_position/velocity_local',
            self.drone_velocity_callback, qos_profile)
        
        # Create service clients
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
        
        # Create publishers
        self.position_pub = self.create_publisher(
            PoseStamped, f'/{namespace}/setpoint_position/local', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for arming service for {namespace}...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for set mode service for {namespace}...")
            
        # Initialize drone states
        self.current_state = State()
        self.drone_state = DroneState(namespace)
        self.setpoint_state = DroneState(f"{namespace}_setpoint")
        self.home_state = DroneState(f"{namespace}_home")
        
        # Set initial positions
        self.setpoint_state.position = np.array([0.0, 0.0, 2.0])
        self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.home_state.position = np.array([0.0, 0.0, 2.0])
        self.home_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Initialize other variables
        self.mission_state = "IDLE"
        self.state_start_time = 0.0
        self.last_setpoint_time = self.get_clock().now()
        
        # Create timers
        self.setpoint_timer = self.create_timer(0.1, self.publish_setpoint, callback_group=self.timer_group)
        self.mission_timer = self.create_timer(0.1, self.mission_step, callback_group=self.timer_group)
        self.demo_timer = self.create_timer(1.0, self.update_demo_orientation, callback_group=self.timer_group)
        
        self.get_logger().info(f"{namespace} initialized - Simple orientation control demo")

    def update_demo_orientation(self):
        """Update demonstration orientation values - MODIFY THIS FOR DIFFERENT BEHAVIORS"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.mission_state == "MISSION":
            # DEMO SEQUENCE - Change these for different behaviors
            
            # Example 1: Static orientation
            # self.demo_roll = 15.0    # 15 degrees right bank
            # self.demo_pitch = 5.0    # 5 degrees nose up
            # self.demo_yaw = 30.0     # 30 degrees right turn
            
            # Example 2: Sinusoidal rolling motion
            self.demo_roll = 20.0 * math.sin(current_time * 0.5)  # Roll back and forth
            self.demo_pitch = 0.0
            self.demo_yaw = 0.0
            
            # Example 3: Circular motion in attitude
            # self.demo_roll = 15.0 * math.sin(current_time * 0.3)
            # self.demo_pitch = 10.0 * math.cos(current_time * 0.3)
            # self.demo_yaw = 30.0 * math.sin(current_time * 0.2)
            
            # Example 4: Step changes every 10 seconds
            # step = int(current_time / 10) % 4
            # orientations = [(0, 0, 0), (20, 0, 0), (0, 15, 0), (0, 0, 45)]
            # self.demo_roll, self.demo_pitch, self.demo_yaw = orientations[step]
            
            # Log current target
            if int(current_time) % 5 == 0:  # Every 5 seconds
                self.get_logger().info(
                    f"Target orientation: Roll={self.demo_roll:.1f}° "
                    f"Pitch={self.demo_pitch:.1f}° Yaw={self.demo_yaw:.1f}°"
                )

    def set_manual_orientation(self, roll_deg, pitch_deg, yaw_deg):
        """Manually set target orientation (call this from external script if needed)"""
        self.demo_roll = roll_deg
        self.demo_pitch = pitch_deg
        self.demo_yaw = yaw_deg
        self.get_logger().info(f"Manual orientation set: R={roll_deg}° P={pitch_deg}° Y={yaw_deg}°")

    def drone_pose_callback(self, msg):
        """Callback for our own position updates"""
        self.drone_state.update_from_pose(msg)

    def drone_velocity_callback(self, msg):
        """Callback for our own velocity updates"""
        self.drone_state.update_from_twist(msg)

    def state_callback(self, msg):
        """Callback for our own PX4 state updates"""
        self.current_state = msg

    def publish_setpoint(self):
        """Timer callback to publish setpoint with orientation control"""
        now = self.get_clock().now()
        
        # Safety check: Detect abnormal position jumps
        if self.position_safety_check:
            current_pos = self.drone_state.position
            setpoint_pos = self.setpoint_state.position
            
            dist_to_setpoint = np.linalg.norm(current_pos - setpoint_pos)
            
            if dist_to_setpoint > self.max_position_jump:
                self.get_logger().warn(f"Safety limiter: Reducing step size from {dist_to_setpoint:.2f}m to {self.max_position_jump:.2f}m")
                direction = (setpoint_pos - current_pos) / dist_to_setpoint
                self.setpoint_state.position = current_pos + direction * self.max_position_jump
        
        # Set target position (hovering at demo altitude)
        self.setpoint_state.position[0] = 0.0  # X position
        self.setpoint_state.position[1] = 0.0  # Y position  
        self.setpoint_state.position[2] = self.demo_altitude  # Z position (altitude)
        
        # ORIENTATION CONTROL - This is where the magic happens
        if self.use_orientation_control:
            # Convert degrees to radians
            target_roll_rad = math.radians(self.demo_roll)
            target_pitch_rad = math.radians(self.demo_pitch)
            target_yaw_rad = math.radians(self.demo_yaw)
            
            # Set target orientation
            desired_orientation = self.setpoint_state.set_target_orientation(
                target_roll_rad, target_pitch_rad, target_yaw_rad
            )
            
            # Apply smooth transition
            if self.smooth_orientation:
                self.setpoint_state.orientation = (
                    self.setpoint_state.orientation * self.orientation_smoothing +
                    desired_orientation * (1.0 - self.orientation_smoothing)
                )
                
                # Normalize the orientation quaternion
                norm = np.linalg.norm(self.setpoint_state.orientation)
                if norm > 0:
                    self.setpoint_state.orientation = self.setpoint_state.orientation / norm
            else:
                self.setpoint_state.orientation = desired_orientation
            
            # Log actual vs target occasionally
            if int(self.get_clock().now().nanoseconds / 1e9 * 10) % 20 == 0:  # Every 2 seconds
                current_roll, current_pitch, current_yaw = self.drone_state.quaternion_to_euler(self.drone_state.orientation)
                target_roll, target_pitch, target_yaw = self.setpoint_state.quaternion_to_euler(self.setpoint_state.orientation)
                
                self.get_logger().info(
                    f"Actual: R={math.degrees(current_roll):.1f}° P={math.degrees(current_pitch):.1f}° Y={math.degrees(current_yaw):.1f}° | "
                    f"Target: R={math.degrees(target_roll):.1f}° P={math.degrees(target_pitch):.1f}° Y={math.degrees(target_yaw):.1f}°"
                )
        
        # Create and publish the pose message
        pose_msg = self.setpoint_state.to_pose_stamped(now.to_msg())
        self.position_pub.publish(pose_msg)
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
            else:
                self.get_logger().warn(f"{self.namespace} Failed to set mode {requested_mode}")
                self.set_mode_async(requested_mode)
        except Exception as e:
            self.get_logger().error(f"Set mode service call failed: {e}")
    
    def start_mission(self):
        """Start the mission state machine"""
        self.mission_state = "INIT"
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(f"{self.namespace} Starting orientation demo mission...")
    
    def return_to_home(self):
        """Initiate return to home procedure"""
        self.mission_state = "RETURN_HOME"
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        self.setpoint_state.position = np.copy(self.drone_state.position)
        self.get_logger().info(f"{self.namespace} Returning to home position...")
    
    def mission_step(self):
        """Mission state machine timer callback"""
        if self.mission_state == "IDLE":
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.state_start_time
        
        if self.mission_state == "INIT":
            if elapsed > 2.0:
                self.mission_state = "ARMING"
                self.get_logger().info(f"{self.namespace} Initiating arming sequence")
                self.arm_async()
        
        elif self.mission_state == "ARMING":
            pass  # Handled by arm_done_callback
            
        elif self.mission_state == "SET_OFFBOARD":
            self.set_mode_async("OFFBOARD")
            
        elif self.mission_state == "TAKEOFF":
            self.setpoint_state.position[2] = self.demo_altitude
            
            if self.use_orientation_control:
                # Level orientation for takeoff
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
            
            if elapsed > 5.0:
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting orientation demo")
        
        elif self.mission_state == "MISSION":
            # Demo is running - orientation updates happen in update_demo_orientation()
            if elapsed > 60.0:  # 1 minute demo
                self.mission_state = "RETURN_HOME"
                self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            # Set setpoint to home position
            self.setpoint_state.position = np.copy(self.home_state.position)
            
            # Level out orientation for return
            if self.use_orientation_control:
                level_orientation = self.setpoint_state.set_target_orientation(0.0, 0.0, 0.0)
                
                if self.smooth_orientation:
                    self.setpoint_state.orientation = (
                        self.setpoint_state.orientation * self.orientation_smoothing +
                        level_orientation * (1.0 - self.orientation_smoothing)
                    )
                    norm = np.linalg.norm(self.setpoint_state.orientation)
                    if norm > 0:
                        self.setpoint_state.orientation = self.setpoint_state.orientation / norm
                else:
                    self.setpoint_state.orientation = level_orientation
            
            # Check if we've reached home
            distance = np.linalg.norm(self.drone_state.position - self.home_state.position)
            
            if distance < 0.5 or elapsed > 10.0:
                self.mission_state = "LANDING"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            if self.use_orientation_control:
                # Level orientation for landing
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
                
            if elapsed > 10.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")

def main(args=None):
    rclpy.init(args=args)
    
    # Create single drone instance
    drone = SimpleDroneControl("drone1")
    
    # Create executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(drone)
    
    try:
        print("\n================================================================")
        print("SIMPLE DRONE ORIENTATION CONTROL DEMO")
        print("================================================================")
        print("Single drone with roll, pitch, yaw control")
        print("- Modify update_demo_orientation() for different behaviors")
        print("- Current demo: Sinusoidal rolling motion")
        print("- Call drone.set_manual_orientation(roll, pitch, yaw) for manual control")
        print("================================================================")
        
        # Start mission
        time.sleep(1.0)
        drone.start_mission()
        
        print("\n✅ Demo active - drone will perform orientation maneuvers!")
        print("💡 Watch the logs for target vs actual orientation values")
        print("🔧 Modify the update_demo_orientation() method to change behavior")
        
        # Example of manual control (uncomment to test)
        # time.sleep(10.0)
        # drone.set_manual_orientation(30.0, 10.0, 45.0)  # 30° roll, 10° pitch, 45° yaw
        
        # Spin until mission complete
        while not drone.mission_completed:
            executor.spin_once()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n🚨 EMERGENCY STOP - Landing drone")
        drone.return_to_home()
        
        # Monitor return sequence
        end_time = time.time() + 10.0
        while time.time() < end_time:
            executor.spin_once()
            time.sleep(0.01)
            
        drone.set_mode_async("AUTO.LAND")
        time.sleep(3.0)
        
    finally:
        print("Shutting down drone control...")
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()