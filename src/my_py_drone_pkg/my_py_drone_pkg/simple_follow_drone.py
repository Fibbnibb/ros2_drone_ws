#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Vector3, TwistStamped, Quaternion
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
import numpy as np


class DroneState:
    """Class to store and handle drone position, orientation, and velocity data"""
    def __init__(self, drone_id):
        self.drone_id = drone_id
        
        # Position data
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w] quaternion
        
        # Velocity data
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]
        self.speed = 0.0  # Scalar speed
        
        # Tracking data
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
        
        self.angular_velocity[0] = twist_msg.twist.angular.x
        self.angular_velocity[1] = twist_msg.twist.angular.y
        self.angular_velocity[2] = twist_msg.twist.angular.z
        
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
        # Convert quaternion to heading vector (forward direction)
        # This is a simplified conversion that works for our use case
        x = 2 * (quat[0] * quat[2] + quat[3] * quat[1])
        y = 2 * (quat[1] * quat[2] - quat[3] * quat[0])
        z = 1 - 2 * (quat[0] * quat[0] + quat[1] * quat[1])
        
        # Project to XY plane and normalize
        heading = np.array([x, y, 0.0])
        norm = np.linalg.norm(heading)
        if norm > 0.001:
            heading = heading / norm
            return heading
        
        return np.array([1.0, 0.0, 0.0])  # Default forward direction
    
    def calculate_orientation_for_velocity(self, velocity_vector):
        """Calculate appropriate orientation quaternion based on velocity direction"""
        # Only change orientation if we have meaningful velocity
        if np.linalg.norm(velocity_vector[:2]) < 0.1:
            return self.orientation  # Keep current orientation if barely moving
            
        # Get the heading angle in the XY plane
        yaw = math.atan2(velocity_vector[1], velocity_vector[0])
        
        # Create quaternion for pure yaw rotation (no roll or pitch)
        # This ensures the drone stays level while changing direction
        qw = math.cos(yaw / 2)
        qz = math.sin(yaw / 2)
        
        # Return quaternion in [x, y, z, w] format with no roll or pitch
        # This is critical to prevent flipping
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
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.is_leader = is_leader
        self.mission_completed = False
        
        # Boids parameters (for followers only)
        self.min_separation = min_separation  # Minimum separation distance in meters (2 feet ≈ 0.61m)
        self.max_speed = 10.0                  # Maximum speed when following (m/s)
        self.current_speed = 6.0              # Current target speed (m/s) - can be changed dynamically
        self.leader_id = "drone1"             # ID of the leader drone to follow
        
        # Boids algorithm weights
        self.separation_weight = 1.7          # Weight for separation rule
        self.alignment_weight = 1.2           # Weight for alignment rule  
        self.cohesion_weight = 0.8            # Weight for cohesion rule
        self.leader_attraction_weight = 1.1   # Weight for following the leader
        
        # Orientation control parameters
        self.use_orientation_control = True   # Enable/disable orientation control
        self.smooth_orientation = True        # Smooth orientation changes to prevent flipping
        self.orientation_smoothing = 0.9      # Increased smoothing (0.9 instead of 0.8)
        self.max_tilt_angle = 15.0           # Maximum tilt angle in degrees
        
        # Safety parameters
        self.position_safety_check = True     # Enable position safety monitoring
        self.max_position_jump = 2.0          # Maximum allowed position change in one step (meters)
        self.max_allowed_distance = 10.0      # Maximum allowed distance from path before emergency stop
        
        # Perception radius - how far drones can "see" other drones
        self.perception_radius = 5.0          # Meters
        
        # Create callback groups - one for timer, one for services
        self.timer_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriptions
        self.state_sub = self.create_subscription(
            State, f'/{namespace}/state', self.state_callback, 10)
        
        self.drone_pose_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.drone_pose_callback, qos_profile)
            
        # Velocity subscription - to know our own velocity
        self.drone_vel_sub = self.create_subscription(
            TwistStamped, f'/{namespace}/local_position/velocity_local',
            self.drone_velocity_callback, qos_profile)
        
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
        
        # Position and velocity sharing publisher 
        self.shared_state_pub = self.create_publisher(
            TwistStamped, f'/shared_states/{namespace}', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for arming service for {namespace}...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for set mode service for {namespace}...")
            
        # Initialize PX4 state
        self.current_state = State()
        
        # Initialize drone state objects
        self.drone_state = DroneState(namespace)
        self.setpoint_state = DroneState(f"{namespace}_setpoint")
        
        # Set initial setpoint
        self.setpoint_state.position = np.array([0.0, 0.0, 2.0])
        self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Store home position
        self.home_state = DroneState(f"{namespace}_home")
        self.home_state.position = np.array([0.0, 0.0, 2.0])
        self.home_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Dictionary to store states of other drones
        self.other_drones = {}
        
        # Subscribe to all shared states
        self.shared_state_subs = []
        
        # Mission state variables
        self.mission_state = "IDLE"
        self.waypoint_index = 0
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
        
        # Share state timer
        self.share_state_timer = self.create_timer(
            0.1,
            self.share_state,
            callback_group=self.timer_group
        )
        
        # Set a higher speed for the leader
        if is_leader:
            self.current_speed = 6.0  # 1.0 m/s for leader
            self.get_logger().info(f"Leader will use constant speed of {self.current_speed} m/s")
        
        # Leader's waypoints (only used if this drone is the leader)
        self.waypoints = []
        if self.is_leader:
            # Define the leader's path
            self.waypoints = [
                {'x': 0.0, 'y': 0.0, 'z': 2.0},      # Takeoff position
                {'x': 5.0, 'y': 0.0, 'z': 2.0},      # Move forward
                {'x': 5.0, 'y': 5.0, 'z': 2.0},      # Move right
                {'x': 0.0, 'y': 5.0, 'z': 2.0},      # Move back
                {'x': 0.0, 'y': 0.0, 'z': 2.0}       # Return home
            ]
        
        self.get_logger().info(f"{namespace} initialized as {'leader' if is_leader else 'follower'}")
        if not is_leader:
            self.get_logger().info(f"Following leader with min separation: {self.min_separation}m")
            self.get_logger().info(f"Boids weights - Separation: {self.separation_weight}, Alignment: {self.alignment_weight}, Cohesion: {self.cohesion_weight}")
        self.get_logger().info(f"Initial speed: {self.current_speed}m/s")
        self.get_logger().info(f"Orientation control: {'Enabled' if self.use_orientation_control else 'Disabled'}")

    def share_state(self):
        """Share this drone's position and velocity with the swarm"""
        # Create shared state message from our DroneState object
        state_msg = self.drone_state.to_shared_state(self.get_clock().now().to_msg())
        self.shared_state_pub.publish(state_msg)

    def register_drone(self, drone_namespace):
        """Register another drone to monitor its state"""
        if drone_namespace == self.namespace:
            return  # Don't register ourselves
            
        # Only create subscription if we don't already have one
        if drone_namespace not in self.other_drones:
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
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Update the drone state from the message
        if drone_namespace in self.other_drones:
            drone_state = self.other_drones[drone_namespace]
            drone_state.update_from_shared_state(msg)
            drone_state.last_update_time = current_time

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
        """Timer callback to publish setpoint"""
        now = self.get_clock().now()
        
        # Safety check: Detect abnormal position jumps
        if self.position_safety_check:
            current_pos = self.drone_state.position
            setpoint_pos = self.setpoint_state.position
            
            # Calculate distance between current position and setpoint
            dist_to_setpoint = np.linalg.norm(current_pos - setpoint_pos)
            
            # If the setpoint is too far from current position, limit it
            if dist_to_setpoint > self.max_position_jump:
                self.get_logger().warn(f"Safety limiter: Reducing step size from {dist_to_setpoint:.2f}m to {self.max_position_jump:.2f}m")
                
                # Calculate direction and limit the distance
                direction = (setpoint_pos - current_pos) / dist_to_setpoint
                self.setpoint_state.position = current_pos + direction * self.max_position_jump
        
        # If orientation control is enabled, calculate proper orientation for current motion
        if self.use_orientation_control:
            # Calculate velocity vector (where we're heading)
            vel_vector = self.setpoint_state.position - self.drone_state.position
            
            # Only update orientation if we're moving significantly
            if np.linalg.norm(vel_vector) > 0.1:
                # Calculate desired orientation for our current velocity
                desired_orientation = self.drone_state.calculate_orientation_for_velocity(vel_vector)
                
                # Apply smooth transition to prevent flipping
                if self.smooth_orientation:
                    # Gradually blend between current and desired orientation
                    # Using stronger smoothing to make changes very gradual
                    self.setpoint_state.orientation = (
                        self.setpoint_state.orientation * self.orientation_smoothing +
                        desired_orientation * (1.0 - self.orientation_smoothing)
                    )
                    
                    # Normalize the orientation quaternion
                    norm = np.linalg.norm(self.setpoint_state.orientation)
                    if norm > 0:
                        self.setpoint_state.orientation = self.setpoint_state.orientation / norm
                else:
                    # Direct assignment without smoothing
                    self.setpoint_state.orientation = desired_orientation
        
        # Create and publish the pose message
        pose_msg = self.setpoint_state.to_pose_stamped(now.to_msg())
        self.position_pub.publish(pose_msg)
        self.last_setpoint_time = now
        
        # If we're a follower, apply the Boids algorithm
        if not self.is_leader and self.mission_state in ["TAKEOFF", "MISSION", "RETURN_HOME"]:
            self.apply_boids_algorithm()

    def apply_boids_algorithm(self):
        """Apply the three Boids rules: separation, alignment, and cohesion"""
        # Initialize steering vectors
        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        leader_attraction = np.zeros(3)
        
        # Count nearby drones
        nearby_drones = 0
        leader_found = False
        
        # Check all other drones
        current_time = self.get_clock().now().nanoseconds / 1e9
        for drone_id, drone_state in self.other_drones.items():
            # Skip if data is too old (more than 1 second)
            if current_time - drone_state.last_update_time > 1.0:
                continue
                
            # Calculate distance to this drone
            distance = self.drone_state.get_distance_to(drone_state)
            
            # Special handling for leader
            if drone_id == self.leader_id:
                leader_found = True
                
                # Always calculate leader attraction regardless of distance
                # Steer towards leader but maintain minimum separation
                if distance > self.min_separation:
                    # Simply steer towards leader
                    leader_attraction = drone_state.position - self.drone_state.position
                else:
                    # Too close to leader, maintain minimum distance
                    # Move away from leader to maintain exactly min_separation
                    if distance > 0:  # Avoid division by zero
                        away_direction = self.drone_state.get_direction_to(drone_state) * -1
                        # Calculate how much further we need to be
                        separation_diff = self.min_separation - distance
                        leader_attraction = away_direction * separation_diff
            
            # Apply standard boids rules only for drones within perception radius
            if distance < self.perception_radius:
                # Skip the leader for standard rules - it only affects leader_attraction
                if drone_id == self.leader_id:
                    continue
                    
                nearby_drones += 1
                
                # 1. Separation: steer to avoid crowding local flockmates
                if distance < self.min_separation and distance > 0:
                    # Calculate separation vector (stronger the closer they are)
                    avoidance = self.drone_state.get_direction_to(drone_state) * -1
                    separation += avoidance * (self.min_separation - distance)
                
                # 2. Alignment: steer towards the average heading of local flockmates
                drone_heading = drone_state.get_heading()
                if np.linalg.norm(drone_heading) > 0:
                    alignment += drone_heading
                
                # 3. Cohesion: steer to move toward the average position of local flockmates
                cohesion += drone_state.position - self.drone_state.position
        
        # Normalize and apply weights to steering vectors
        if nearby_drones > 0:
            # Normalize alignment by number of drones
            alignment /= nearby_drones
            # Normalize cohesion by number of drones and apply weight
            cohesion /= nearby_drones
        
        # Calculate the combined steering force
        steering = np.zeros(3)
        
        # Apply separation force (always applied even if no nearby drones)
        steering += separation * self.separation_weight
        
        # Apply other forces only if we have nearby drones
        if nearby_drones > 0:
            # Apply alignment force
            steering += alignment * self.alignment_weight
            # Apply cohesion force
            steering += cohesion * self.cohesion_weight
            
        # Apply leader attraction (if leader was found)
        if leader_found:
            steering += leader_attraction * self.leader_attraction_weight
            
            # Try to match leader's altitude
            leader_state = self.other_drones[self.leader_id]
            steering[2] = (leader_state.position[2] - self.drone_state.position[2]) * 2.0  # Stronger altitude adjustment
        
        # Calculate new velocity by adding steering to current velocity
        new_vel = self.drone_state.velocity + steering
        
        # Limit speed
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = (new_vel / speed) * self.max_speed
            
        # Calculate new position
        new_pos = self.drone_state.position + new_vel * 0.1  # 0.1 is the timer interval
        
        # Update setpoint - position is updated here, orientation in publish_setpoint
        self.setpoint_state.position = new_pos
        
        # Log the applied forces for debugging (only occasionally to reduce spam)
        if leader_found and int(current_time * 10) % 50 == 0:
            self.get_logger().debug(
                f"Boids forces - Separation: {np.linalg.norm(separation):.2f}, "
                f"Alignment: {np.linalg.norm(alignment):.2f}, "
                f"Cohesion: {np.linalg.norm(cohesion):.2f}, "
                f"Leader: {np.linalg.norm(leader_attraction):.2f}"
            )

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
    
    def start_mission(self):
        """Start the mission state machine"""
        self.get_logger().info(f"{self.namespace} Starting mission...")
        self.mission_state = "INIT"
        self.waypoint_index = 0
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
    
    def return_to_home(self):
        """Initiate return to home procedure"""
        self.get_logger().info(f"{self.namespace} Returning to home position...")
        
        if self.is_leader:
            # For leader, set the last waypoint (which should be home)
            self.waypoint_index = len(self.waypoints) - 1
            self.mission_state = "MISSION"
        else:
            # For followers, transition to RETURN_HOME state
            self.mission_state = "RETURN_HOME"
            
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Reset setpoint to current position first to avoid sudden jumps
        self.setpoint_state.position = np.copy(self.drone_state.position)
    
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
            self.setpoint_state.position[2] = 2.0
            
            # Keep orientation stable during takeoff
            if self.use_orientation_control:
                # Use default level orientation for takeoff (no roll or pitch)
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
            
            # After 5 seconds, or when altitude is reached, transition to mission
            if elapsed > 5.0:
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting mission")
        
        elif self.mission_state == "MISSION":
            if self.is_leader:
                # Leader follows waypoints with constant speed
                if self.waypoint_index < len(self.waypoints):
                    # Skip the first waypoint (takeoff position) as we're already there
                    if self.waypoint_index == 0 and elapsed > 1.0:
                        self.waypoint_index = 1
                        self.get_logger().info(f"{self.namespace} Starting to follow waypoints")
                        next_wp = self.waypoints[self.waypoint_index]
                        self.get_logger().info(f"{self.namespace} Heading to waypoint 1 at ({next_wp['x']}, {next_wp['y']}, {next_wp['z']})")
                        return
                    # Get current waypoint
                    wp = self.waypoints[self.waypoint_index]
                    
                    # Get current position
                    current_pos = self.drone_state.position
                    wp_pos = np.array([wp['x'], wp['y'], wp['z']])
                    
                    # Calculate distance and direction to waypoint
                    direction = wp_pos - current_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0:
                        direction = direction / distance  # Normalize direction
                    
                    # Calculate the step size - use a more conservative approach
                    # Move at moderate speed and ensure we don't make sudden jumps
                    step_size = min(self.current_speed * 0.15, distance * 0.3)  # Reduced values
                    
                    # Only slow down when very close to the waypoint
                    if distance < 1.0:
                        step_size = min(step_size, distance * 0.3)
                    
                    # Update setpoint to move towards the waypoint
                    if distance > 0.1:  # Only move if not already at target point
                        # Calculate new position 
                        new_position = current_pos + direction * step_size
                        
                        # Safety check: Verify the new position isn't too far from the path
                        if self.position_safety_check:
                            # Vector from start to destination
                            start_pos = np.array([0, 0, 2.0])  # Home position
                            end_pos = wp_pos
                            path_vector = end_pos - start_pos
                            path_length = np.linalg.norm(path_vector)
                            
                            if path_length > 0:
                                # Project current position onto path
                                path_direction = path_vector / path_length
                                t = np.dot(current_pos - start_pos, path_direction)
                                closest_point = start_pos + t * path_direction
                                
                                # Calculate distance from path
                                distance_from_path = np.linalg.norm(current_pos - closest_point)
                                
                                # If too far from path, log warning and limit movement
                                if distance_from_path > self.max_allowed_distance:
                                    self.get_logger().warn(f"Position safety: Drone has deviated {distance_from_path:.2f}m from path! Limiting movement.")
                                    # Adjust setpoint to move back toward path
                                    correction_vector = closest_point - current_pos
                                    correction_distance = np.linalg.norm(correction_vector)
                                    if correction_distance > 0:
                                        correction_direction = correction_vector / correction_distance
                                        new_position = current_pos + correction_direction * step_size
                        
                        # Update the setpoint
                        self.setpoint_state.position = new_position
                        self.setpoint_state.position[2] = wp['z']  # Direct altitude control
                        
                        # Update orientation to face the direction of travel if orientation control enabled
                        if self.use_orientation_control:
                            # Calculate desired orientation for travel direction
                            desired_orientation = self.drone_state.calculate_orientation_for_velocity(direction)
                            
                            # Apply smooth transition to prevent flipping
                            if self.smooth_orientation:
                                # Gradually blend between current and desired orientation
                                self.setpoint_state.orientation = (
                                    self.setpoint_state.orientation * self.orientation_smoothing +
                                    desired_orientation * (1.0 - self.orientation_smoothing)
                                )
                                
                                # Normalize the orientation quaternion
                                norm = np.linalg.norm(self.setpoint_state.orientation)
                                if norm > 0:
                                    self.setpoint_state.orientation = self.setpoint_state.orientation / norm
                            else:
                                # Direct assignment without smoothing
                                self.setpoint_state.orientation = desired_orientation
                        
                        # Log info for debugging
                        if self.waypoint_index < len(self.waypoints) and int(current_time) % 5 == 0:  # Log every 5 seconds
                            self.get_logger().info(
                                f"Moving to waypoint {self.waypoint_index}: " +
                                f"Current: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}), " +
                                f"Target: ({wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f}), " +
                                f"Distance: {distance:.2f}m"
                            )
                    
                    # Check if we've reached the waypoint
                    if distance < 0.5:  # Within 0.5m of the waypoint
                        self.waypoint_index += 1
                        if self.waypoint_index < len(self.waypoints):
                            next_wp = self.waypoints[self.waypoint_index]
                            self.get_logger().info(f"{self.namespace} Reached waypoint {self.waypoint_index}, heading to next waypoint at ({next_wp['x']}, {next_wp['y']}, {next_wp['z']})")
                        else:
                            self.get_logger().info(f"{self.namespace} Reached final waypoint")
                            
                        self.state_start_time = current_time  # Reset timer for next waypoint
                        
                        # If we've reached the final waypoint, transition to landing
                        if self.waypoint_index >= len(self.waypoints):
                            self.mission_state = "LANDING"
                            self.state_start_time = current_time
                            self.get_logger().info(f"{self.namespace} Mission complete, initiating landing")
                            self.set_mode_async("AUTO.LAND")
                else:
                    # If we somehow get here, transition to landing
                    self.mission_state = "LANDING"
                    self.state_start_time = current_time
                    self.set_mode_async("AUTO.LAND")
            else:
                # Followers use the Boids algorithm (handled in publish_setpoint)
                
                # After a timeout, assume mission is complete
                if elapsed > 180.0:  # 3 minutes max mission time
                    self.mission_state = "RETURN_HOME"
                    self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            # For the leader, this is handled in the mission state
            # For followers, we need special handling
            if not self.is_leader:
                # Set setpoint to home position
                self.setpoint_state.position = np.copy(self.home_state.position)
                
                # Set orientation to face home
                if self.use_orientation_control:
                    # Calculate direction to home
                    home_direction = self.home_state.position - self.drone_state.position
                    if np.linalg.norm(home_direction) > 0.1:
                        # Calculate orientation to face home
                        desired_orientation = self.drone_state.calculate_orientation_for_velocity(home_direction)
                        
                        # Apply smooth transition
                        if self.smooth_orientation:
                            # Gradually blend between current and desired orientation
                            self.setpoint_state.orientation = (
                                self.setpoint_state.orientation * self.orientation_smoothing +
                                desired_orientation * (1.0 - self.orientation_smoothing)
                            )
                            
                            # Normalize the orientation quaternion
                            norm = np.linalg.norm(self.setpoint_state.orientation)
                            if norm > 0:
                                self.setpoint_state.orientation = self.setpoint_state.orientation / norm
                        else:
                            # Direct assignment without smoothing
                            self.setpoint_state.orientation = desired_orientation
                
                # Check if we've reached home
                distance = np.linalg.norm(self.drone_state.position - self.home_state.position)
                
                if distance < 0.5 or elapsed > 10.0:  # Within 0.5m of home or 10 seconds
                    self.mission_state = "LANDING"
                    self.state_start_time = current_time
                    self.get_logger().info(f"{self.namespace} Return complete, initiating landing")
                    self.set_mode_async("AUTO.LAND")
        
        elif self.mission_state == "LANDING":
            # Keep orientation level during landing to prevent flipping
            if self.use_orientation_control:
                # Level orientation for landing (no roll or pitch)
                self.setpoint_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
                
            # Wait for landing to complete
            if elapsed > 10.0:
                self.mission_state = "COMPLETE"
                self.mission_completed = True
                self.get_logger().info(f"{self.namespace} Mission completed")


def main(args=None):
    rclpy.init(args=args)
    
    # Create drone instances - one leader and multiple followers
    # Min separation distance is set to 0.61 meters (2 feet)
    leader = BoidsDroneSwarm("drone1", is_leader=True)
    follower1 = BoidsDroneSwarm("drone2", is_leader=False, min_separation=0.61)
    follower2 = BoidsDroneSwarm("drone3", is_leader=False, min_separation=0.61)
    
    # Register drones with each other
    leader.register_drone("drone2")
    leader.register_drone("drone3")
    
    follower1.register_drone("drone1")
    follower1.register_drone("drone3")
    
    follower2.register_drone("drone1")
    follower2.register_drone("drone2")
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(leader)
    executor.add_node(follower1)
    executor.add_node(follower2)
    
    try:
        # Start missions with a small delay between each
        time.sleep(1.0)
        leader.start_mission()
        time.sleep(1.0)
        follower1.start_mission()
        time.sleep(1.0)
        follower2.start_mission()
        
        # Spin to allow missions to progress
        all_completed = False
        while not all_completed:
            executor.spin_once()
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            # Check if all drones have completed their missions
            all_completed = (
                leader.mission_completed and 
                follower1.mission_completed and 
                follower2.mission_completed
            )
            
    except KeyboardInterrupt:
        # Return to home for all drones if interrupted
        leader.return_to_home()
        follower1.return_to_home()
        follower2.return_to_home()
        
        # Give some time for return to begin
        time.sleep(2.0)
        
        # Continue processing until we decide to exit
        end_time = time.time() + 10.0  # Process for up to 10 more seconds
        while time.time() < end_time:
            executor.spin_once()
            time.sleep(0.01)
            
        # Finally land the drones
        leader.set_mode_async("AUTO.LAND")
        follower1.set_mode_async("AUTO.LAND")
        follower2.set_mode_async("AUTO.LAND")
        time.sleep(3.0)
        
    finally:
        # Cleanup
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()