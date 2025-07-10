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
        self.speed = 0.0  # Scalar speed
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
        
        # Calculate scalar speed (magnitude of velocity)
        self.speed = np.linalg.norm(self.velocity[:2])  # Using only x and y for ground speed
        
    def update_from_shared_state(self, state_msg):
        """Update from the custom shared state message format"""
        self.position[0] = state_msg.twist.linear.x
        self.position[1] = state_msg.twist.linear.y
        self.position[2] = state_msg.twist.linear.z
        
        # Velocity data from the hijacked angular part
        self.velocity[0] = state_msg.twist.angular.x
        self.velocity[1] = state_msg.twist.angular.y
        self.speed = state_msg.twist.angular.z
        
    def get_distance_to(self, other_state):
        """Calculate distance to another drone"""
        return np.linalg.norm(self.position - other_state.position)
    
    def get_direction_to(self, other_state):
        """Get normalized direction vector to another drone"""
        direction = other_state.position - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            return direction / distance
        return np.zeros(3)
    
    def get_heading(self):
        """Get heading direction as a unit vector in the XY plane"""
        if np.linalg.norm(self.velocity[:2]) > 0.1:  # If moving fast enough
            heading = self.velocity[:2] / np.linalg.norm(self.velocity[:2])
            return np.array([heading[0], heading[1], 0.0])
        
        # If not moving, extract heading from orientation quaternion
        quat = self.orientation
        x = 2 * (quat[0] * quat[2] + quat[3] * quat[1])
        y = 2 * (quat[1] * quat[2] - quat[3] * quat[0])
        
        # Project to XY plane and normalize
        heading = np.array([x, y, 0.0])
        norm = np.linalg.norm(heading)
        if norm > 0.001:
            return heading / norm
        
        return np.array([1.0, 0.0, 0.0])  # Default forward direction
    
    def calculate_orientation_for_velocity(self, velocity_vector):
        """Calculate appropriate orientation quaternion based on velocity direction"""
        # Only change orientation if we have meaningful velocity
        if np.linalg.norm(velocity_vector[:2]) < 0.1:
            return self.orientation
            
        # Get the heading angle in the XY plane
        yaw = math.atan2(velocity_vector[1], velocity_vector[0])
        
        # Create quaternion for pure yaw rotation (no roll or pitch)
        qw = math.cos(yaw / 2)
        qz = math.sin(yaw / 2)
        
        return np.array([0.0, 0.0, qz, qw])
    
    # NEW METHODS FOR ORIENTATION FOLLOWING
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
    
    def follow_leader_orientation(self, leader_state, roll_weight=0.8, pitch_weight=0.7, yaw_weight=0.9):
        """Simple method to follow leader's orientation"""
        if not leader_state:
            return self.orientation
        
        # Get leader's angles
        leader_roll, leader_pitch, leader_yaw = self.quaternion_to_euler(leader_state.orientation)
        
        # Get current angles
        current_roll, current_pitch, current_yaw = self.quaternion_to_euler(self.orientation)
        
        # Calculate target angles (blend with leader)
        target_roll = leader_roll * roll_weight + current_roll * (1 - roll_weight)
        target_pitch = leader_pitch * pitch_weight + current_pitch * (1 - pitch_weight)
        target_yaw = leader_yaw * yaw_weight + current_yaw * (1 - yaw_weight)
        
        # Apply safety limits
        max_roll = math.radians(45)   # 45 degrees
        max_pitch = math.radians(30)  # 30 degrees
        
        target_roll = np.clip(target_roll, -max_roll, max_roll)
        target_pitch = np.clip(target_pitch, -max_pitch, max_pitch)
        
        # Convert back to quaternion
        return self.euler_to_quaternion(target_roll, target_pitch, target_yaw)
    
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
    
    def to_shared_state(self, stamp=None):
        """Convert to shared state message (TwistStamped)"""
        msg = TwistStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = self.drone_id
        
        # Include position in linear part
        msg.twist.linear.x = self.position[0]
        msg.twist.linear.y = self.position[1]
        msg.twist.linear.z = self.position[2]
        
        # Include velocity in angular part
        msg.twist.angular.x = self.velocity[0]
        msg.twist.angular.y = self.velocity[1]
        msg.twist.angular.z = self.speed
        
        return msg

class BoidsDroneSwarm(Node):
    """Node to control a drone in a Boids-style swarm with leader-follower dynamics"""
    def __init__(self, namespace, is_leader=False, min_separation=0.61):
        super().__init__(f"drone_node_{namespace}")
        
        # Basic parameters
        self.namespace = namespace
        self.is_leader = is_leader
        self.mission_completed = False
        
        # Boids parameters (for followers only)
        self.min_separation = min_separation
        self.max_speed = 20.0
        self.current_speed = 12.0
        self.leader_id = "drone1"
        
        # Weights for Boids algorithm
        self.separation_weight = 1.4
        self.alignment_weight = 1.4
        self.cohesion_weight = 0.9
        self.leader_attraction_weight = 1.5
        
        # Dispersion parameters
        self.dispersion_weight = 2.5
        self.dispersion_radius = 7.0
        self.regrouping_factor = 0.5
        
        # Control parameters
        self.use_orientation_control = True
        self.follow_leader_orientation = True  # NEW: Enable orientation following
        self.smooth_orientation = True
        self.orientation_smoothing = 0.7
        
        # Safety parameters
        self.position_safety_check = True
        self.max_position_jump = 5.5
        self.perception_radius = 7.0
        
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
        
        self.shared_state_pub = self.create_publisher(
            TwistStamped, f'/shared_states/{namespace}', 10)
        
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
        self.other_drones = {}
        self.shared_state_subs = []
        self.mission_state = "IDLE"
        self.waypoint_index = 0
        self.state_start_time = 0.0
        self.last_setpoint_time = self.get_clock().now()
        
        # Create timers
        self.setpoint_timer = self.create_timer(0.1, self.publish_setpoint, callback_group=self.timer_group)
        self.mission_timer = self.create_timer(0.1, self.mission_step, callback_group=self.timer_group)
        self.share_state_timer = self.create_timer(0.1, self.share_state, callback_group=self.timer_group)
        
        # Log initialization info
        log_msg = f"{namespace} initialized as {'leader (manual control)' if is_leader else 'follower'}"
        if not is_leader:
            log_msg += f", min separation: {self.min_separation}m, orientation following: {self.follow_leader_orientation}"
        self.get_logger().info(log_msg)

    def toggle_orientation_following(self, enable=True):
        """Enable/disable orientation following"""
        self.follow_leader_orientation = enable
        if enable:
            self.get_logger().info(f"{self.namespace} Full orientation following ENABLED")
        else:
            self.get_logger().info(f"{self.namespace} Full orientation following DISABLED")

    def share_state(self):
        """Share this drone's position and velocity with the swarm"""
        state_msg = self.drone_state.to_shared_state(self.get_clock().now().to_msg())
        self.shared_state_pub.publish(state_msg)

    def register_drone(self, drone_namespace):
        """Register another drone to monitor its state"""
        if drone_namespace == self.namespace or drone_namespace in self.other_drones:
            return
            
        self.get_logger().info(f"Registering drone: {drone_namespace}")
        
        # Create subscription for this drone's state
        sub = self.create_subscription(
            TwistStamped,
            f'/shared_states/{drone_namespace}',
            lambda msg, ns=drone_namespace: self.other_drone_state_callback(msg, ns),
            10
        )
        self.shared_state_subs.append(sub)
        
        # Create DroneState object for this drone
        self.other_drones[drone_namespace] = DroneState(drone_namespace)

    def other_drone_state_callback(self, msg, drone_namespace):
        """Callback for receiving state updates from other drones"""
        if drone_namespace in self.other_drones:
            drone_state = self.other_drones[drone_namespace]
            drone_state.update_from_shared_state(msg)
            drone_state.last_update_time = self.get_clock().now().nanoseconds / 1e9

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
        """Timer callback to publish setpoint with enhanced orientation control"""
        now = self.get_clock().now()
        
        # For leader drone, skip publishing setpoints
        if self.is_leader:
            self.last_setpoint_time = now
            return
        
        # Safety check: Detect abnormal position jumps
        if self.position_safety_check:
            current_pos = self.drone_state.position
            setpoint_pos = self.setpoint_state.position
            
            dist_to_setpoint = np.linalg.norm(current_pos - setpoint_pos)
            
            if dist_to_setpoint > self.max_position_jump:
                self.get_logger().warn(f"Safety limiter: Reducing step size from {dist_to_setpoint:.2f}m to {self.max_position_jump:.2f}m")
                direction = (setpoint_pos - current_pos) / dist_to_setpoint
                self.setpoint_state.position = current_pos + direction * self.max_position_jump
        
        # ENHANCED ORIENTATION CONTROL
        if self.use_orientation_control:
            leader_state = None
            
            # Check if we have fresh leader data
            if self.leader_id in self.other_drones:
                current_time = self.get_clock().now().nanoseconds / 1e9
                leader_state = self.other_drones[self.leader_id]
                
                # Check if leader data is fresh (within 1 second)
                if current_time - leader_state.last_update_time > 1.0:
                    leader_state = None
            
            # Calculate desired orientation
            if leader_state and self.follow_leader_orientation:
                # NEW: Follow leader's full orientation (roll, pitch, yaw)
                desired_orientation = self.drone_state.follow_leader_orientation(
                    leader_state, 
                    roll_weight=0.8,   # 80% of leader's roll
                    pitch_weight=0.7,  # 70% of leader's pitch
                    yaw_weight= 0.6     # 60% of leader's yaw
                )
                
                # Optional: Log orientation following occasionally
                if int(current_time * 10) % 50 == 0:  # Every 5 seconds
                    leader_roll, leader_pitch, leader_yaw = self.drone_state.quaternion_to_euler(leader_state.orientation)
                    my_roll, my_pitch, my_yaw = self.drone_state.quaternion_to_euler(desired_orientation)
                    self.get_logger().info(
                        f"Following leader - "
                        f"Leader: R={math.degrees(leader_roll):.1f}° P={math.degrees(leader_pitch):.1f}° Y={math.degrees(leader_yaw):.1f}° | "
                        f"Target: R={math.degrees(my_roll):.1f}° P={math.degrees(my_pitch):.1f}° Y={math.degrees(my_yaw):.1f}°"
                    )
            else:
                # Fall back to velocity-based orientation (original behavior)
                vel_vector = self.setpoint_state.position - self.drone_state.position
                if np.linalg.norm(vel_vector) > 0.1:
                    desired_orientation = self.drone_state.calculate_orientation_for_velocity(vel_vector)
                else:
                    desired_orientation = self.setpoint_state.orientation
            
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
        
        # Create and publish the pose message
        pose_msg = self.setpoint_state.to_pose_stamped(now.to_msg())
        self.position_pub.publish(pose_msg)
        self.last_setpoint_time = now
        
        # If we're a follower in an active mission state, apply the Boids algorithm
        if not self.is_leader and self.mission_state in ["TAKEOFF", "MISSION", "RETURN_HOME"]:
            self.apply_boids_algorithm()

    def apply_boids_algorithm(self):
        """Apply the Boids rules: separation, alignment, cohesion, and dispersion/regrouping"""
        # Initialize vectors
        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        leader_attraction = np.zeros(3)
        dispersion = np.zeros(3)
        
        nearby_drones = 0
        leader_found = False
        leader_state = None
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Calculate center of mass of follower drones
        center_of_mass = np.zeros(3)
        follower_count = 0
        
        # First pass: identify leader and calculate center of mass
        for drone_id, drone_state in self.other_drones.items():
            if current_time - drone_state.last_update_time > 1.0:
                continue
            
            if drone_id == self.leader_id:
                leader_found = True
                leader_state = drone_state
                continue
            
            center_of_mass += drone_state.position
            follower_count += 1
        
        # Finalize center of mass calculation
        if follower_count > 0:
            center_of_mass /= follower_count
            swarm_center = (center_of_mass * follower_count + self.drone_state.position) / (follower_count + 1)
        else:
            swarm_center = np.copy(self.drone_state.position)
        
        # Determine if leader is approaching the swarm center
        leader_approaching = False
        dispersion_strength = 0.0
        
        if leader_found and follower_count > 0:
            leader_to_center_dist = np.linalg.norm(leader_state.position - swarm_center)
            leader_velocity = leader_state.velocity
            leader_direction = swarm_center - leader_state.position
            
            if np.linalg.norm(leader_direction) > 0:
                leader_direction = leader_direction / np.linalg.norm(leader_direction)
                
            leader_approach_factor = np.dot(leader_velocity, leader_direction)
            
            dispersion_radius = self.perception_radius * 2
            if leader_approach_factor > 0.5 and leader_to_center_dist < dispersion_radius:
                leader_approaching = True
                dispersion_strength = 1.0 - (leader_to_center_dist / dispersion_radius)
                self.last_dispersion_strength = dispersion_strength
                
                if not hasattr(self, 'previous_dispersion_strength') or abs(dispersion_strength - self.previous_dispersion_strength) > 0.2:
                    self.get_logger().info(f"Leader approaching! Dispersion active: {dispersion_strength:.2f}")
                    self.previous_dispersion_strength = dispersion_strength
            else:
                self.last_dispersion_strength = 0.0
                
                if hasattr(self, 'previous_dispersion_strength') and self.previous_dispersion_strength > 0.2:
                    self.get_logger().info(f"Dispersion deactivated, regrouping")
                    self.previous_dispersion_strength = 0.0
                    self.dispersion_end_time = current_time
        
        # Second pass: process all other drones for standard boids rules
        for drone_id, drone_state in self.other_drones.items():
            if current_time - drone_state.last_update_time > 1.0:
                continue
                
            distance = self.drone_state.get_distance_to(drone_state)
            
            # Special handling for leader
            if drone_id == self.leader_id:
                if distance > self.min_separation:
                    attraction_modifier = 1.0 - (dispersion_strength * 0.8)
                    leader_attraction = (drone_state.position - self.drone_state.position) * attraction_modifier
                else:
                    if distance > 0:
                        away_direction = self.drone_state.get_direction_to(drone_state) * -1
                        separation_diff = self.min_separation - distance
                        leader_attraction = away_direction * separation_diff
                
                # Calculate dispersion vector if leader is approaching
                if leader_approaching:
                    away_from_leader = self.drone_state.position - leader_state.position
                    
                    if np.linalg.norm(away_from_leader) > 0:
                        away_from_leader = away_from_leader / np.linalg.norm(away_from_leader)
                        
                    leader_vel = leader_state.velocity
                    if np.linalg.norm(leader_vel) > 0:
                        leader_vel = leader_vel / np.linalg.norm(leader_vel)
                        
                        perp_vector = np.array([leader_vel[1], -leader_vel[0], 0.0])
                        
                        if hash(self.namespace) % 2 == 0:
                            perp_vector = -perp_vector
                        
                        dispersion_vector = away_from_leader * 0.6 + perp_vector * 0.4
                        dispersion += dispersion_vector * dispersion_strength * 2.0
            
            # Apply standard boids rules for drones within perception radius
            if distance < self.perception_radius:
                if drone_id == self.leader_id:
                    continue
                    
                nearby_drones += 1
                
                # 1. Separation
                if distance < self.min_separation or distance > 0:
                    avoidance = self.drone_state.get_direction_to(drone_state) * -1
                    separation += avoidance * (self.min_separation - distance)
                
                # 2. Alignment
                drone_heading = drone_state.get_heading()
                if np.linalg.norm(drone_heading) > 0:
                    alignment += drone_heading
                
                # 3. Cohesion
                cohesion_modifier = 1.0 - (dispersion_strength * 0.9)
                cohesion += (drone_state.position - self.drone_state.position) * cohesion_modifier
        
        # Process the combined forces
        if nearby_drones > 0:
            alignment /= nearby_drones
            cohesion /= nearby_drones
        
        # Calculate the combined steering force
        steering = separation * self.separation_weight
        
        if nearby_drones > 0:
            steering += alignment * self.alignment_weight
            
            effective_cohesion_weight = self.cohesion_weight
            
            if hasattr(self, 'previous_dispersion_strength') and self.previous_dispersion_strength > 0.2 and dispersion_strength < 0.1:
                if not hasattr(self, 'dispersion_end_time'):
                    self.dispersion_end_time = current_time
                
                time_since_dispersion = current_time - self.dispersion_end_time
                if time_since_dispersion < 5.0:
                    cohesion_boost = 1.0 + (1.5 * (1.0 - time_since_dispersion/5.0))
                    effective_cohesion_weight *= cohesion_boost
                    
                    if int(current_time * 10) % 20 == 0:
                        self.get_logger().info(f"Regrouping: Cohesion boost {cohesion_boost:.2f}x")
            
            steering += cohesion * effective_cohesion_weight
        
        if leader_found:
            steering += leader_attraction * self.leader_attraction_weight
            
            if leader_approaching:
                dispersion_weight = 2.5
                steering += dispersion * dispersion_weight
            
            # Try to match leader's altitude
            leader_state = self.other_drones[self.leader_id]
            steering[2] = (leader_state.position[2] - self.drone_state.position[2]) * 2.0
        
        # Calculate new velocity and limit speed
        new_vel = self.drone_state.velocity + steering
        
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = (new_vel / speed) * self.max_speed
            
        # Calculate new position and update setpoint
        new_pos = self.drone_state.position + new_vel * 0.15
        self.setpoint_state.position = new_pos

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
                    if not self.is_leader:
                        self.mission_state = "TAKEOFF"
                        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.get_logger().warn(f"{self.namespace} Failed to set mode {requested_mode}")
                self.set_mode_async(requested_mode)
        except Exception as e:
            self.get_logger().error(f"Set mode service call failed: {e}")
    
    def start_mission(self):
        """Start the mission state machine"""
        if self.is_leader:
            self.mission_state = "LEADER_MANUAL"
            self.get_logger().info(f"{self.namespace} Ready for manual control via QGroundControl")
            self.arm_async()
        else:
            self.mission_state = "INIT"
            self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f"{self.namespace} Starting mission...")
    
    def return_to_home(self):
        """Initiate return to home procedure"""
        if self.is_leader:
            self.get_logger().info(f"{self.namespace} Leader is manually controlled - please return to home using QGroundControl")
        else:
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
            
        if self.mission_state == "LEADER_MANUAL":
            if int(current_time) % 5 == 0:
                current_pos = self.drone_state.position
                self.get_logger().info(
                    f"Leader position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})"
                )
        
        elif self.mission_state == "INIT":
            if elapsed > 2.0:
                self.mission_state = "ARMING"
                self.get_logger().info(f"{self.namespace} Initiating arming sequence")
                self.arm_async()
        
        elif self.mission_state == "ARMING":
            pass
            
        elif self.mission_state == "SET_OFFBOARD":
            if not self.is_leader:
                self.set_mode_async("OFFBOARD")
            
        elif self.mission_state == "TAKEOFF":
            self.setpoint_state.position[2] = 2.0
            
            if self.use_orientation_control:
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
            
            if elapsed > 5.0:
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting mission")
        
        elif self.mission_state == "MISSION":
            if self.is_leader:
                pass
            else:
                if int(current_time) % 5 == 0:
                    if hasattr(self, 'last_dispersion_strength') and self.last_dispersion_strength > 0.2:
                        self.get_logger().info(
                            f"{self.namespace} Dispersion active: {self.last_dispersion_strength:.2f}, "
                            f"Position: ({self.drone_state.position[0]:.2f}, {self.drone_state.position[1]:.2f}, {self.drone_state.position[2]:.2f})"
                        )
                    
                if elapsed > 180.0:
                    self.mission_state = "RETURN_HOME"
                    self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            if not self.is_leader:
                self.setpoint_state.position = np.copy(self.home_state.position)
                
                if self.use_orientation_control:
                    home_direction = self.home_state.position - self.drone_state.position
                    if np.linalg.norm(home_direction) > 0.1:
                        desired_orientation = self.drone_state.calculate_orientation_for_velocity(home_direction)
                        
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
                
                distance = np.linalg.norm(self.drone_state.position - self.home_state.position)
                
                if distance < 0.5 or elapsed > 10.0:
                    self.mission_state = "LANDING"
                    self.state_start_time = current_time
                    self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                    self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            if self.use_orientation_control and not self.is_leader:
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
                
            if elapsed > 10.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")

def main(args=None):
    rclpy.init(args=args)
    
    # Create drone instances
    leader = BoidsDroneSwarm("drone1", is_leader=True, min_separation=0.61)
    follower1 = BoidsDroneSwarm("drone2", is_leader=False, min_separation=0.61)
    follower2 = BoidsDroneSwarm("drone3", is_leader=False, min_separation=0.61)
    follower3 = BoidsDroneSwarm("drone4", is_leader=False, min_separation=0.61)
    
    # Configure orientation following - you can customize these weights
    follower1.toggle_orientation_following(True)
    follower2.toggle_orientation_following(True)
    follower3.toggle_orientation_following(True)
    
    # Register drones with each other
    leader.register_drone("drone2")
    leader.register_drone("drone3")
    leader.register_drone("drone4")

    follower1.register_drone("drone1")
    follower1.register_drone("drone3")
    follower1.register_drone("drone4")

    follower2.register_drone("drone1")
    follower2.register_drone("drone2")
    follower2.register_drone("drone4")

    follower3.register_drone("drone1")
    follower3.register_drone("drone2")
    follower3.register_drone("drone3")
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(leader)
    executor.add_node(follower1)
    executor.add_node(follower2)
    executor.add_node(follower3)
    
    try:
        print("\n================================================================")
        print("DRONE SWARM WITH SIMPLE ORIENTATION FOLLOWING")
        print("================================================================")
        print("Leader drone (drone1): Manual control via QGroundControl")
        print("Follower drones (drone2-4): Follow leader's ROLL, PITCH, and YAW")
        print("- Roll weight: 80% | Pitch weight: 70% | Yaw weight: 60%")
        print("- Safety limits: ±45° roll, ±30° pitch")
        print("- Smooth orientation transitions enabled")
        print("================================================================")
        
        # Start missions
        time.sleep(1.0)
        leader.start_mission()
        time.sleep(3.0)
        print("Starting follower drones...")
        follower1.start_mission()
        time.sleep(1.0)
        follower2.start_mission()
        time.sleep(1.0)
        follower3.start_mission()
        
        print("\n Swarm active - followers will match leader's orientation!")
        print(" Try rolling, pitching, or yawing the leader to see followers respond")
        print(" To adjust weights, modify the roll_weight, pitch_weight, yaw_weight in publish_setpoint()")
        
        # Spin to allow missions to progress
        all_completed = False
        while not all_completed:
            executor.spin_once()
            time.sleep(0.01)
            
            all_completed = (
                follower1.mission_completed and 
                follower2.mission_completed and
                follower3.mission_completed
            )
            
    except KeyboardInterrupt:
        print("\n INITIATING EMERGENCY RETURN TO HOME")
        follower1.return_to_home()
        follower2.return_to_home()
        follower3.return_to_home()
        
        print("Leader drone should be manually returned to home using QGroundControl")
        
        # Monitor return sequence
        end_time = time.time() + 10.0
        while time.time() < end_time:
            executor.spin_once()
            time.sleep(0.005)
            
        # Land the follower drones
        print("Landing follower drones")
        follower1.set_mode_async("AUTO.LAND")
        follower2.set_mode_async("AUTO.LAND")
        follower3.set_mode_async("AUTO.LAND")
        print("Please land the leader drone using QGroundControl")
        time.sleep(3.0)
        
    finally:
        print("Shutting down drone swarm nodes...")
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()