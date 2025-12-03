#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    
    def vector_to_quaternion(self, velocity_vector):
        """Convert velocity vector to quaternion orientation (for boid following)"""
        if np.linalg.norm(velocity_vector) < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0])  # Default orientation
            
        # Normalize the velocity vector to get forward direction
        forward = velocity_vector / np.linalg.norm(velocity_vector)
        
        # World up vector
        world_up = np.array([0.0, 0.0, 1.0])
        
        # Calculate right vector (cross product of world_up and forward)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            # Forward is parallel to world_up, use arbitrary right vector
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        
        # Calculate up vector (cross product of forward and right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix [right, up, forward]
        rot_matrix = np.column_stack((right, up, forward))
        
        # Convert to quaternion using scipy
        rotation = R.from_matrix(rot_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        
        return quat
    
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
    
    def set_target_orientation_from_vector(self, velocity_vector):
        """Set target orientation from velocity vector (for boid following)"""
        self.orientation = self.vector_to_quaternion(velocity_vector)
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

class AttitudeDroneControl(Node):
    """Enhanced drone control with AttitudeTarget for full 3D attitude control"""
    def __init__(self, namespace):
        super().__init__(f"attitude_drone_{namespace}")
        
        # Basic parameters
        self.namespace = namespace
        self.mission_completed = False
        
        # Control parameters
        self.use_attitude_control = True  # Use AttitudeTarget instead of PoseStamped
        self.use_vector_control = True    # Use velocity vector for orientation
        self.smooth_orientation = True
        self.orientation_smoothing = 0.1  # Slower for smoother transitions
        
        # Altitude control parameters
        self.altitude_kp = 0.3
        self.altitude_ki = 0.05
        self.altitude_kd = 0.1
        self.altitude_integral = 0.0
        self.altitude_prev_error = 0.0
        self.base_thrust = 0.6
        
        # Demonstration parameters - THESE ARE THE VALUES YOU CAN CHANGE
        self.demo_roll = 0.0      # Target roll in degrees
        self.demo_pitch = 0.0     # Target pitch in degrees  
        self.demo_yaw = 0.0       # Target yaw in degrees
        self.demo_altitude = 2.0  # Target altitude in meters
        
        # Boid simulation parameters (for demonstration)
        self.boid_velocity = np.array([0.0, 0.0, 0.0])  # Current boid velocity
        self.boid_speed = 2.0     # Desired boid speed
        
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
        
        # NEW: AttitudeTarget publisher for full 3D attitude control
        self.attitude_pub = self.create_publisher(
            AttitudeTarget, f'/{namespace}/setpoint_raw/attitude', 10)
        
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
        self.setpoint_timer = self.create_timer(0.05, self.publish_setpoint, callback_group=self.timer_group)  # 20Hz
        self.mission_timer = self.create_timer(0.1, self.mission_step, callback_group=self.timer_group)
        self.demo_timer = self.create_timer(0.5, self.update_demo_behavior, callback_group=self.timer_group)
        
        self.get_logger().info(f"{namespace} initialized - Enhanced attitude control with AttitudeTarget")

    def update_demo_behavior(self):
        """Update demonstration behavior - MODIFY THIS FOR DIFFERENT BEHAVIORS"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.mission_state == "MISSION":
            if self.use_vector_control:
                # DEMO: Simulate boid-like circular motion
                radius = 3.0
                angular_speed = 0.3
                
                # Circular path in XY plane
                center_x, center_y = 0.0, 0.0
                target_x = center_x + radius * math.cos(current_time * angular_speed)
                target_y = center_y + radius * math.sin(current_time * angular_speed)
                
                # Calculate velocity vector pointing toward next position
                next_time = current_time + 0.1
                next_x = center_x + radius * math.cos(next_time * angular_speed)
                next_y = center_y + radius * math.sin(next_time * angular_speed)
                
                # Velocity vector
                self.boid_velocity = np.array([
                    (next_x - target_x) / 0.1,
                    (next_y - target_y) / 0.1,
                    0.0  # No vertical component
                ])
                
                # Normalize and scale to desired speed
                if np.linalg.norm(self.boid_velocity) > 0:
                    self.boid_velocity = self.boid_velocity / np.linalg.norm(self.boid_velocity) * self.boid_speed
                
                # Add some vertical oscillation for more interesting movement
                self.boid_velocity[2] = 0.5 * math.sin(current_time * 0.5)
                
                # Log velocity occasionally
                if int(current_time * 2) % 10 == 0:  # Every 5 seconds
                    self.get_logger().info(
                        f"Boid velocity: [{self.boid_velocity[0]:.2f}, {self.boid_velocity[1]:.2f}, {self.boid_velocity[2]:.2f}]"
                    )
                    
            else:
                # Traditional Euler angle control
                self.demo_roll = 20.0 * math.sin(current_time * 0.5)
                self.demo_pitch = 10.0 * math.cos(current_time * 0.3)
                self.demo_yaw = 30.0 * math.sin(current_time * 0.2)

    def set_boid_velocity(self, velocity_vector):
        """Set boid velocity for vector-based control"""
        self.boid_velocity = np.array(velocity_vector)
        self.get_logger().info(f"Boid velocity set: [{velocity_vector[0]:.2f}, {velocity_vector[1]:.2f}, {velocity_vector[2]:.2f}]")

    def drone_pose_callback(self, msg):
        """Callback for our own position updates"""
        self.drone_state.update_from_pose(msg)

    def drone_velocity_callback(self, msg):
        """Callback for our own velocity updates"""
        self.drone_state.update_from_twist(msg)

    def state_callback(self, msg):
        """Callback for our own PX4 state updates"""
        self.current_state = msg

    def compute_altitude_thrust(self):
        """Compute thrust command for altitude control using PID"""
        current_altitude = self.drone_state.position[2]
        target_altitude = self.demo_altitude
        
        # PID controller for altitude
        error = target_altitude - current_altitude
        self.altitude_integral += error * 0.05  # dt = 0.05s (20Hz)
        self.altitude_integral = np.clip(self.altitude_integral, -1.0, 1.0)  # Anti-windup
        
        derivative = (error - self.altitude_prev_error) / 0.05
        self.altitude_prev_error = error
        
        thrust_adjustment = (self.altitude_kp * error + 
                           self.altitude_ki * self.altitude_integral + 
                           self.altitude_kd * derivative)
        
        thrust = self.base_thrust + thrust_adjustment
        return np.clip(thrust, 0.0, 1.0)

    def publish_setpoint(self):
        """Timer callback to publish setpoint with attitude control"""
        now = self.get_clock().now()
        
        if self.use_attitude_control:
            # === ATTITUDE TARGET CONTROL ===
            att = AttitudeTarget()
            att.header.stamp = now.to_msg()
            att.header.frame_id = "base_link"
            
            # Mask off body rates; we're only sending orientation+thrust
            att.type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE |
                AttitudeTarget.IGNORE_PITCH_RATE |
                AttitudeTarget.IGNORE_YAW_RATE
            )
            
            # Determine target orientation
            if self.use_vector_control and np.linalg.norm(self.boid_velocity) > 0.1:
                # Use velocity vector to determine orientation
                desired_orientation = self.setpoint_state.set_target_orientation_from_vector(self.boid_velocity)
            else:
                # Use Euler angles
                target_roll_rad = math.radians(self.demo_roll)
                target_pitch_rad = math.radians(self.demo_pitch)
                target_yaw_rad = math.radians(self.demo_yaw)
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
            
            # Set quaternion orientation
            qx, qy, qz, qw = self.setpoint_state.orientation
            att.orientation.x = float(qx)
            att.orientation.y = float(qy)
            att.orientation.z = float(qz)
            att.orientation.w = float(qw)
            
            # Compute thrust for altitude control
            att.thrust = float(self.compute_altitude_thrust())
            
            # Publish attitude target
            self.attitude_pub.publish(att)
            
            # Log occasionally
            if int(now.nanoseconds / 1e9 * 4) % 40 == 0:  # Every 10 seconds at 4Hz
                current_roll, current_pitch, current_yaw = self.drone_state.quaternion_to_euler(self.drone_state.orientation)
                target_roll, target_pitch, target_yaw = self.setpoint_state.quaternion_to_euler(self.setpoint_state.orientation)
                
                self.get_logger().info(
                    f"Attitude | Current: R={math.degrees(current_roll):.1f}° P={math.degrees(current_pitch):.1f}° Y={math.degrees(current_yaw):.1f}° | "
                    f"Target: R={math.degrees(target_roll):.1f}° P={math.degrees(target_pitch):.1f}° Y={math.degrees(target_yaw):.1f}° | "
                    f"Thrust: {att.thrust:.2f}"
                )
        else:
            # === POSITION TARGET CONTROL (Original) ===
            # Set target position
            self.setpoint_state.position[0] = 0.0
            self.setpoint_state.position[1] = 0.0
            self.setpoint_state.position[2] = self.demo_altitude
            
            # Traditional orientation control
            if self.use_vector_control and np.linalg.norm(self.boid_velocity) > 0.1:
                desired_orientation = self.setpoint_state.set_target_orientation_from_vector(self.boid_velocity)
            else:
                target_roll_rad = math.radians(self.demo_roll)
                target_pitch_rad = math.radians(self.demo_pitch)
                target_yaw_rad = math.radians(self.demo_yaw)
                desired_orientation = self.setpoint_state.set_target_orientation(
                    target_roll_rad, target_pitch_rad, target_yaw_rad
                )
            
            if self.smooth_orientation:
                self.setpoint_state.orientation = (
                    self.setpoint_state.orientation * self.orientation_smoothing +
                    desired_orientation * (1.0 - self.orientation_smoothing)
                )
                norm = np.linalg.norm(self.setpoint_state.orientation)
                if norm > 0:
                    self.setpoint_state.orientation = self.setpoint_state.orientation / norm
            else:
                self.setpoint_state.orientation = desired_orientation
            
            # Create and publish pose message
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
        self.get_logger().info(f"{self.namespace} Starting enhanced attitude control mission...")
    
    def return_to_home(self):
        """Initiate return to home procedure"""
        self.mission_state = "RETURN_HOME"
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
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
            if elapsed > 8.0:  # Longer takeoff time for attitude control
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting enhanced attitude demo")
        
        elif self.mission_state == "MISSION":
            # Demo is running - behavior updates happen in update_demo_behavior()
            if elapsed > 120.0:  # 2 minute demo
                self.mission_state = "RETURN_HOME"
                self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            # Level out orientation and stop boid movement
            self.boid_velocity = np.array([0.0, 0.0, 0.0])
            self.demo_roll = 0.0
            self.demo_pitch = 0.0
            self.demo_yaw = 0.0
            
            if elapsed > 10.0:
                self.mission_state = "LANDING"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            if elapsed > 15.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")

def main(args=None):
    rclpy.init(args=args)
    
    # Create single drone instance
    drone = AttitudeDroneControl("drone1")
    
    # Create executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(drone)
    
    try:
        print("\n================================================================")
        print("ENHANCED DRONE ATTITUDE CONTROL DEMO")
        print("================================================================")
        print("Features:")
        print("- AttitudeTarget publisher for true 3D attitude control")
        print("- Vector-based orientation from velocity (boid-ready)")
        print("- PID altitude control with thrust commands")
        print("- Smooth orientation transitions")
        print("- Circular motion demo with banking")
        print("================================================================")
        
        # Start mission
        time.sleep(1.0)
        drone.start_mission()
        
        print("\n✅ Enhanced attitude control active!")
        print("🎯 Drone will bank and pitch into turns like a real aircraft")
        print("🔄 Current demo: Circular motion with velocity-based orientation")
        print("🎮 Call drone.set_boid_velocity([vx, vy, vz]) for custom control")
        
        # Example of manual vector control (uncomment to test)
        # time.sleep(15.0)
        # drone.set_boid_velocity([3.0, 1.0, 0.5])  # Forward, right, up
        
        # Spin until mission complete
        while not drone.mission_completed:
            executor.spin_once()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n🚨 EMERGENCY STOP - Landing drone")
        drone.return_to_home()
        
        # Monitor return sequence
        end_time = time.time() + 15.0
        while time.time() < end_time:
            executor.spin_once()
            time.sleep(0.01)
            
        drone.set_mode_async("AUTO.LAND")
        time.sleep(5.0)
        
    finally:
        print("Shutting down enhanced drone control...")
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()