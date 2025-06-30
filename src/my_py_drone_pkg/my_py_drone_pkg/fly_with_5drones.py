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
        # Position data from the hijacked linear part
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
        msg.twist.angular.z = self.speed  # Store scalar speed directly
        
        return msg


class BoidsDroneSwarm(Node):
    def __init__(self, namespace, is_leader=False, min_separation=0.61):
        super().__init__(f"drone_node_{namespace}")
        
        # Basic parameters
        self.namespace = namespace
        self.is_leader = is_leader
        self.mission_completed = False
        
        # Boids parameters (for followers only)
        self.min_separation = min_separation  # Minimum separation distance in meters
        self.max_speed = 20.0                 # Maximum speed when following (m/s)
        self.current_speed = 12.0             # Current target speed (m/s)
        self.leader_id = "drone1"             # ID of the leader drone to follow
        
        # Weights for Boids algorithm
        self.separation_weight = 1.4
        self.alignment_weight = 1.4
        self.cohesion_weight = 0.9
        self.leader_attraction_weight = 1.5
        
        # Dispersion parameters
        self.dispersion_weight = 2.5          # Weight for dispersion (overrides other forces)
        self.dispersion_radius = 7.0          # Radius within which dispersion activates
        self.regrouping_factor = 0.5          # How quickly drones regroup after dispersion
        
        # Control parameters
        self.use_orientation_control = True
        self.smooth_orientation = True
        self.orientation_smoothing = 0.7  # Smoothing factor for orientation (0.0 to 1.0)
        
        # Safety parameters
        self.position_safety_check = True
        self.max_position_jump = 5.5  # Maximum allowed jump in position (m)
        self.perception_radius = 7.0  # Meters
        
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
            log_msg += f", min separation: {self.min_separation}m"
        self.get_logger().info(log_msg)

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

    def is_ready_for_arming(self):
        """Check if drone is in a state where it can be armed"""
        # Check if we've received any state updates
        if not hasattr(self.current_state, 'armed'):
            return False
            
        # Make sure we're not already armed
        if self.current_state.armed:
            return True  # Already armed, so we're good
            
        # Check if we have received any position updates
        has_position = (np.linalg.norm(self.drone_state.position) != 0.0)
        
        return has_position

    def publish_setpoint(self):
        """Timer callback to publish setpoint"""
        now = self.get_clock().now()
        
        # For leader drone, skip publishing setpoints since it will be controlled by QGroundControl
        if self.is_leader:
            self.last_setpoint_time = now
            return
        
        # Safety check: Detect abnormal position jumps
        if self.position_safety_check:
            current_pos = self.drone_state.position
            setpoint_pos = self.setpoint_state.position
            
            # Calculate distance between current position and setpoint
            dist_to_setpoint = np.linalg.norm(current_pos - setpoint_pos)
            
            # If the setpoint is too far from current position, limit it
            if dist_to_setpoint > self.max_position_jump:
                self.get_logger().warn(f"Safety limiter: Reducing step size from {dist_to_setpoint:.2f}m to {self.max_position_jump:.2f}m")
                direction = (setpoint_pos - current_pos) / dist_to_setpoint
                self.setpoint_state.position = current_pos + direction * self.max_position_jump
        
        # Handle orientation control
        if self.use_orientation_control:
            vel_vector = self.setpoint_state.position - self.drone_state.position
            
            if np.linalg.norm(vel_vector) > 0.1:
                desired_orientation = self.drone_state.calculate_orientation_for_velocity(vel_vector)
                
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
        dispersion = np.zeros(3)  # New vector for dispersion behavior
        
        nearby_drones = 0
        leader_found = False
        leader_state = None
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Calculate center of mass of follower drones (excluding ourselves and the leader)
        center_of_mass = np.zeros(3)
        follower_count = 0
        
        # First pass: identify leader and calculate center of mass
        for drone_id, drone_state in self.other_drones.items():
            # Skip if data is too old (more than 1 second)
            if current_time - drone_state.last_update_time > 1.0:
                continue
            
            # Special handling for leader
            if drone_id == self.leader_id:
                leader_found = True
                leader_state = drone_state
                continue
            
            # Add to center of mass calculation (only other followers)
            center_of_mass += drone_state.position
            follower_count += 1
        
        # Finalize center of mass calculation
        if follower_count > 0:
            center_of_mass /= follower_count
            # Include our own position in the swarm's center of mass
            swarm_center = (center_of_mass * follower_count + self.drone_state.position) / (follower_count + 1)
        else:
            # If no other followers detected, just use our position
            swarm_center = np.copy(self.drone_state.position)
        
        # Determine if leader is approaching the swarm center
        leader_approaching = False
        dispersion_strength = 0.0
        
        if leader_found and follower_count > 0:
            # Calculate distance from leader to swarm center
            leader_to_center_dist = np.linalg.norm(leader_state.position - swarm_center)
            
            # Check if leader is moving toward the swarm center
            leader_velocity = leader_state.velocity
            leader_direction = swarm_center - leader_state.position
            
            if np.linalg.norm(leader_direction) > 0:
                leader_direction = leader_direction / np.linalg.norm(leader_direction)
                
            # Dot product > 0 means leader is moving toward center
            leader_approach_factor = np.dot(leader_velocity, leader_direction)
            
            # Dispersion activates when leader is approaching and within dispersion radius
            dispersion_radius = self.perception_radius * 1.5  # Larger than normal perception radius
            if leader_approach_factor > 0.5 and leader_to_center_dist < dispersion_radius:
                leader_approaching = True
                # Strength increases as leader gets closer
                dispersion_strength = 1.0 - (leader_to_center_dist / dispersion_radius)
                # Store for logging
                self.last_dispersion_strength = dispersion_strength
                
                # Log first activation or significant changes
                if not hasattr(self, 'previous_dispersion_strength') or abs(dispersion_strength - self.previous_dispersion_strength) > 0.2:
                    self.get_logger().info(f"Leader approaching! Dispersion active: {dispersion_strength:.2f}")
                    self.previous_dispersion_strength = dispersion_strength
            else:
                # Reset dispersion when not active
                self.last_dispersion_strength = 0.0
                
                # Log deactivation only once and mark end time for regrouping
                if hasattr(self, 'previous_dispersion_strength') and self.previous_dispersion_strength > 0.2:
                    self.get_logger().info(f"Dispersion deactivated, regrouping")
                    self.previous_dispersion_strength = 0.0
                    # Mark the time when dispersion ended for regrouping boost
                    self.dispersion_end_time = current_time
        
        # Second pass: process all other drones for standard boids rules
        for drone_id, drone_state in self.other_drones.items():
            # Skip if data is too old
            if current_time - drone_state.last_update_time > 1.0:
                continue
                
            # Calculate distance to this drone
            distance = self.drone_state.get_distance_to(drone_state)
            
            # Special handling for leader
            if drone_id == self.leader_id:
                if distance > self.min_separation:
                    # Steer towards leader, but reduce attraction if leader is approaching center
                    attraction_modifier = 1.0 - (dispersion_strength * 0.8)
                    leader_attraction = (drone_state.position - self.drone_state.position) * attraction_modifier
                else:
                    # Maintain minimum distance from leader
                    if distance > 0:
                        away_direction = self.drone_state.get_direction_to(drone_state) * -1
                        separation_diff = self.min_separation - distance
                        leader_attraction = away_direction * separation_diff
                
                # Calculate dispersion vector if leader is approaching
                if leader_approaching:
                    # Direction from leader to drone
                    away_from_leader = self.drone_state.position - leader_state.position
                    
                    if np.linalg.norm(away_from_leader) > 0:
                        away_from_leader = away_from_leader / np.linalg.norm(away_from_leader)
                        
                    # Also move perpendicular to leader's velocity
                    leader_vel = leader_state.velocity
                    if np.linalg.norm(leader_vel) > 0:
                        leader_vel = leader_vel / np.linalg.norm(leader_vel)
                        
                        # Calculate perpendicular component (in 2D plane)
                        perp_vector = np.array([leader_vel[1], -leader_vel[0], 0.0])
                        
                        # Randomize direction (left or right perpendicular)
                        if hash(self.namespace) % 2 == 0:  # Deterministic but different per drone
                            perp_vector = -perp_vector
                        
                        # Combine away vector with perpendicular component
                        dispersion_vector = away_from_leader * 0.6 + perp_vector * 0.4
                        dispersion += dispersion_vector * dispersion_strength * 2.0
            
            # Apply standard boids rules for drones within perception radius
            if distance < self.perception_radius:
                # Skip the leader for standard rules - it only affects leader_attraction
                if drone_id == self.leader_id:
                    continue
                    
                nearby_drones += 1
                
                # 1. Separation: steer to avoid crowding local flockmates
                if distance < self.min_separation and distance > 0:
                    avoidance = self.drone_state.get_direction_to(drone_state) * -1
                    separation += avoidance * (self.min_separation - distance)
                
                # 2. Alignment: steer towards the average heading of local flockmates
                drone_heading = drone_state.get_heading()
                if np.linalg.norm(drone_heading) > 0:
                    alignment += drone_heading
                
                # 3. Cohesion: steer to move toward the average position of local flockmates
                # Reduce cohesion if dispersion is active
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
            
            # Dynamic cohesion weight: increase after dispersion to help regroup
            effective_cohesion_weight = self.cohesion_weight
            
            # If dispersion was recently active but now inactive, temporarily boost cohesion
            if hasattr(self, 'previous_dispersion_strength') and self.previous_dispersion_strength > 0.2 and dispersion_strength < 0.1:
                # Calculate time since dispersion ended
                if not hasattr(self, 'dispersion_end_time'):
                    self.dispersion_end_time = current_time
                    
                time_since_dispersion = current_time - self.dispersion_end_time
                if time_since_dispersion < 5.0:  # Apply boosted cohesion for 5 seconds
                    # Boost cohesion to help regroup
                    cohesion_boost = 1.0 + (1.5 * (1.0 - time_since_dispersion/5.0))
                    effective_cohesion_weight *= cohesion_boost
                    
                    if int(current_time * 10) % 20 == 0:  # Log occasionally
                        self.get_logger().info(f"Regrouping: Cohesion boost {cohesion_boost:.2f}x")
            
            steering += cohesion * effective_cohesion_weight
            
        if leader_found:
            steering += leader_attraction * self.leader_attraction_weight
            
            # Add dispersion force if leader is approaching
            if leader_approaching:
                # Dispersion overrides other forces when active
                dispersion_weight = 2.5  # Higher than other weights
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
        new_pos = self.drone_state.position + new_vel * 0.15  # 0.15 is the timer interval
        self.setpoint_state.position = new_pos

    def arm_async(self, retry_count=0, max_retries=5):
        """Non-blocking arm request with exponential backoff"""
        if retry_count >= max_retries:
            self.get_logger().error(f"{self.namespace} Max arming retries reached")
            return
            
        if not self.is_ready_for_arming():
            self.get_logger().warn(f"{self.namespace} Not ready for arming, retrying in 1 second (attempt {retry_count+1}/{max_retries})")
            # Schedule retry
            timer = self.create_timer(
                1.0, 
                lambda: self.arm_async(retry_count + 1, max_retries),
                callback_group=self.timer_group
            )
            timer.cancel_on_destruction = True
            return
        
        # Proceed with arming
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        future.add_done_callback(
            lambda f: self.arm_done_callback(f, retry_count, max_retries)
        )
    
    def arm_done_callback(self, future, retry_count=0, max_retries=5):
        """Callback for arm service response with retry logic"""
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"{self.namespace} Armed successfully")
                if self.mission_state == "ARMING":
                    self.mission_state = "SET_OFFBOARD"
                    self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.get_logger().warn(f"{self.namespace} Arming failed (attempt {retry_count+1}/{max_retries})")
                # Exponential backoff
                backoff_time = 1.0 * (2 ** retry_count)
                self.get_logger().info(f"{self.namespace} Retrying in {backoff_time:.1f} seconds")
                
                # Schedule retry with backoff
                timer = self.create_timer(
                    backoff_time, 
                    lambda: self.arm_async(retry_count + 1, max_retries),
                    callback_group=self.timer_group
                )
                timer.cancel_on_destruction = True  # Auto-cleanup
        except Exception as e:
            self.get_logger().error(f"Arm service call failed: {e}")
    
    def set_mode_async(self, mode, retry_count=0, max_retries=5):
        """Non-blocking mode change request with retry capability"""
        if retry_count >= max_retries:
            self.get_logger().error(f"{self.namespace} Max set_mode retries reached for {mode}")
            return
            
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(
            lambda f: self.mode_done_callback(f, mode, retry_count, max_retries)
        )
    
    def mode_done_callback(self, future, requested_mode, retry_count=0, max_retries=5):
        """Callback for set_mode service response with retry logic"""
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info(f"{self.namespace} Mode {requested_mode} set successfully")
                
                if self.mission_state == "SET_OFFBOARD" and requested_mode == "OFFBOARD":
                    if not self.is_leader:
                        self.mission_state = "TAKEOFF"
                        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.get_logger().warn(f"{self.namespace} Failed to set mode {requested_mode} (attempt {retry_count+1}/{max_retries})")
                # Exponential backoff
                backoff_time = 1.0 * (1.5 ** retry_count)
                self.get_logger().info(f"{self.namespace} Retrying in {backoff_time:.1f} seconds")
                
                # Schedule retry with backoff
                timer = self.create_timer(
                    backoff_time, 
                    lambda: self.set_mode_async(requested_mode, retry_count + 1, max_retries),
                    callback_group=self.timer_group
                )
                timer.cancel_on_destruction = True  # Auto-cleanup
        except Exception as e:
            self.get_logger().error(f"Set mode service call failed: {e}")
    
    def start_mission(self):
        """Start the mission state machine"""
        if self.is_leader:
            # For the leader, just arm and don't set OFFBOARD mode
            self.mission_state = "LEADER_MANUAL"
            self.get_logger().info(f"{self.namespace} Ready for manual control via QGroundControl")
            self.arm_async()
        else:
            # For followers, start the normal mission sequence
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
            self.setpoint_state.position = np.copy(self.drone_state.position)  # Avoid sudden jumps
            self.get_logger().info(f"{self.namespace} Returning to home position...")
    
    def mission_step(self):
        """Mission state machine timer callback"""
        if self.mission_state == "IDLE":
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.state_start_time
            
        if self.mission_state == "LEADER_MANUAL":
            # Special state for the manually controlled leader
            if int(current_time) % 5 == 0:  # Log every 5 seconds
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
            # Handled by arm_done_callback
            pass
            
        elif self.mission_state == "SET_OFFBOARD":
            if not self.is_leader:
                self.set_mode_async("OFFBOARD")
            
        elif self.mission_state == "TAKEOFF":
            self.setpoint_state.position[2] = 2.0
            
            if self.use_orientation_control:
                # Level orientation for takeoff
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
            
            if elapsed > 5.0:
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting mission")
        
        elif self.mission_state == "MISSION":
            # Leader is manually controlled
            if self.is_leader:
                # Just monitor position, handled in LEADER_MANUAL state
                pass
            else:
                # Log swarm status every few seconds
                if int(current_time) % 5 == 0:
                    # Log swarm status
                    if hasattr(self, 'last_dispersion_strength') and self.last_dispersion_strength > 0.2:
                        self.get_logger().info(
                            f"{self.namespace} Dispersion active: {self.last_dispersion_strength:.2f}, "
                            f"Position: ({self.drone_state.position[0]:.2f}, {self.drone_state.position[1]:.2f}, {self.drone_state.position[2]:.2f})"
                        )
                    
                # Followers use Boids algorithm (handled in publish_setpoint)
                if elapsed > 180.0:  # 3 minutes max mission time
                    self.mission_state = "RETURN_HOME"
                    self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            if not self.is_leader:
                # Set setpoint to home position
                self.setpoint_state.position = np.copy(self.home_state.position)
                
                # Handle orientation
                if self.use_orientation_control:
                    home_direction = self.home_state.position - self.drone_state.position
                    if np.linalg.norm(home_direction) > 0.1:
                        desired_orientation = self.drone_state.calculate_orientation_for_velocity(home_direction)
                        
                        if self.smooth_orientation:
                            self.setpoint_state.orientation = (
                                self.setpoint_state.orientation * self.orientation_smoothing +
                                desired_orientation * (1.0 - self.orientation_smoothing)
                            )
                            
                            # Normalize
                            norm = np.linalg.norm(self.setpoint_state.orientation)
                            if norm > 0:
                                self.setpoint_state.orientation = self.setpoint_state.orientation / norm
                        else:
                            self.setpoint_state.orientation = desired_orientation
                
                # Check if we've reached home
                distance = np.linalg.norm(self.drone_state.position - self.home_state.position)
                
                if distance < 0.5 or elapsed > 10.0:
                    self.mission_state = "LANDING"
                    self.state_start_time = current_time
                    self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                    self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            if self.use_orientation_control and not self.is_leader:
                # Level orientation for landing
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
                
            if elapsed > 10.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")


def main(args=None):
    rclpy.init(args=args)
    
    # Create drone instances
    leader = BoidsDroneSwarm("drone1", is_leader=True)
    follower1 = BoidsDroneSwarm("drone2", is_leader=False, min_separation=0.61)
    follower2 = BoidsDroneSwarm("drone3", is_leader=False, min_separation=0.61)
    follower3 = BoidsDroneSwarm("drone4", is_leader=False, min_separation=0.61)
    follower4 = BoidsDroneSwarm("drone5", is_leader=False, min_separation=0.61)
    
    # Register drones with each other - Leader knows about all followers
    leader.register_drone("drone2")
    leader.register_drone("drone3")
    leader.register_drone("drone4")
    leader.register_drone("drone5")

    # Follower 1 knows about leader and all other followers
    follower1.register_drone("drone1")
    follower1.register_drone("drone3")
    follower1.register_drone("drone4")
    follower1.register_drone("drone5")

    # Follower 2 knows about leader and all other followers
    follower2.register_drone("drone1")
    follower2.register_drone("drone2")
    follower2.register_drone("drone4")
    follower2.register_drone("drone5")

    # Follower 3 knows about leader and all other followers
    follower3.register_drone("drone1")
    follower3.register_drone("drone2")
    follower3.register_drone("drone3")
    follower3.register_drone("drone5")
    
    # Follower 4 knows about leader and all other followers
    follower4.register_drone("drone1")
    follower4.register_drone("drone2")
    follower4.register_drone("drone3")
    follower4.register_drone("drone4")
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=10)  # Adjusted thread count for 5 drones
    executor.add_node(leader)
    executor.add_node(follower1)
    executor.add_node(follower2)
    executor.add_node(follower3)
    executor.add_node(follower4)
    
    try:
        print("\n================================================================")
        print("DRONE SWARM SETUP WITH MANUAL LEADER CONTROL")
        print("================================================================")
        print("Leader drone (drone1): Will be controlled manually via QGroundControl")
        print("Follower drones (drone2-drone5): Will follow using Boids algorithm")
        
        # Start missions with longer delays between drones
        time.sleep(1.0)
        leader.start_mission()
        time.sleep(7.0)  # Increased from 5.0
        print("Starting follower drones...")
        follower1.start_mission()
        time.sleep(3.0)  # Increased from 1.0
        follower2.start_mission()
        time.sleep(3.0)  # Increased from 1.0
        follower3.start_mission()
        time.sleep(3.0)  # Increased from 1.0
        follower4.start_mission()
        
        # Spin to allow missions to progress
        print("Swarm active - follower drones will now respond to leader movement")
        all_completed = False
        while not all_completed:
            executor.spin_once()
            time.sleep(0.005)  # Reduced sleep time for better responsiveness
            
            # For leader drone, we don't rely on mission_completed since it's manually controlled
            all_completed = (
                follower1.mission_completed and 
                follower2.mission_completed and
                follower3.mission_completed and
                follower4.mission_completed
            )
            
    except KeyboardInterrupt:
        print("\nINITIATING EMERGENCY RETURN TO HOME")
        # Return followers to home
        follower1.return_to_home()
        follower2.return_to_home()
        follower3.return_to_home()
        follower4.return_to_home()
        
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
        follower4.set_mode_async("AUTO.LAND")
        print("Please land the leader drone using QGroundControl")
        time.sleep(3.0)
        
    finally:
        print("Shutting down drone swarm nodes...")
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()