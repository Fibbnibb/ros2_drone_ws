#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
import numpy as np

class PIDController:
    """Simple PID controller implementation"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits  # Tuple (min, max)
        
        self.reset()
        
    def reset(self):
        """Reset the controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, setpoint, process_variable):
        """Update the controller with current values"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Avoid division by zero
        if dt <= 0:
            return 0.0
            
        # Calculate error
        error = setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.output_limits is not None:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Store values for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output

class FormationDrone(Node):
    def __init__(self, namespace, is_leader=False, formation_distance=2.0):
        super().__init__(f"drone_node_{namespace}")
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.is_leader = is_leader
        self.mission_completed = False
        self.formation_distance = formation_distance
        
        # Formation parameters
        self.formation_offset = {'x': 0, 'y': 0, 'z': 0}  # Default offset from leader
        
        # Separation parameters
        self.separation_threshold = 0.1  # Reduced to 0.1 meters (10cm)
        self.separation_weight = 1.5    # Increased weight for stronger reaction at close distances
        
        # Create PID controllers for position control with increased gains
        self.pid_x = PIDController(kp=0.8, ki=0.15, kd=0.25, output_limits=(-2.0, 2.0))
        self.pid_y = PIDController(kp=0.8, ki=0.15, kd=0.25, output_limits=(-2.0, 2.0))
        self.pid_z = PIDController(kp=0.8, ki=0.15, kd=0.25, output_limits=(-1.5, 1.5))
        
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
        
        # Position sharing publisher (for other drones to know where we are)
        self.shared_pose_pub = self.create_publisher(
            PoseStamped, f'/shared_poses/{namespace}', 10)
        
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
        
        # Dictionary to store positions of other drones
        self.other_drones = {}
        
        # Subscribe to all shared poses
        # Create subscriptions for all drones in the swarm
        self.shared_pose_subs = []
        
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
        
        # Share position timer
        self.share_position_timer = self.create_timer(
            0.1,
            self.share_position,
            callback_group=self.timer_group
        )
        
        # Leader's waypoints (only used if this drone is the leader)
        self.waypoints = []
        if self.is_leader:
            # Define the leader's path
            self.waypoints = [
                {'x': 0.0, 'y': 0.0, 'z': 2.0},      # Takeoff
                {'x': 5.0, 'y': 0.0, 'z': 2.0},      # Move forward
                {'x': 5.0, 'y': 5.0, 'z': 2.0},      # Move right
                {'x': 0.0, 'y': 5.0, 'z': 2.0},      # Move back
                {'x': 0.0, 'y': 0.0, 'z': 2.0}       # Return home
            ]
        
        self.get_logger().info(f"{namespace} initialized as {'leader' if is_leader else 'follower'}")

    def share_position(self):
        """Share this drone's position with the swarm"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.namespace
        pose_msg.pose = self.current_pose.pose
        self.shared_pose_pub.publish(pose_msg)

    def register_drone(self, drone_namespace):
        """Register another drone to monitor its position"""
        if drone_namespace == self.namespace:
            return  # Don't register ourselves
            
        # Only create subscription if we don't already have one
        if drone_namespace not in self.other_drones:
            self.get_logger().info(f"Registering drone: {drone_namespace}")
            
            # Create subscription for this drone
            sub = self.create_subscription(
                PoseStamped,
                f'/shared_poses/{drone_namespace}',
                lambda msg, ns=drone_namespace: self.other_drone_pose_callback(msg, ns),
                10
            )
            self.shared_pose_subs.append(sub)
            self.other_drones[drone_namespace] = {
                'pose': PoseStamped(),
                'last_update': 0.0
            }

    def set_formation_offset(self, x_offset=0.0, y_offset=0.0, z_offset=0.0):
        """Set this drone's formation offset from the leader"""
        self.formation_offset = {'x': x_offset, 'y': y_offset, 'z': z_offset}
        self.get_logger().info(f"Formation offset set to: x={x_offset}, y={y_offset}, z={z_offset}")

    def other_drone_pose_callback(self, msg, drone_namespace):
        """Callback for receiving position updates from other drones"""
        self.other_drones[drone_namespace]['pose'] = msg
        self.other_drones[drone_namespace]['last_update'] = self.get_clock().now().nanoseconds / 1e9

    def drone_pose_callback(self, msg):
        """Callback for our own position updates"""
        self.current_pose = msg

    def state_callback(self, msg):
        """Callback for our own state updates"""
        self.current_state = msg

    def publish_setpoint(self):
        """Timer callback to publish setpoint"""
        now = self.get_clock().now()
        self.setpoint.header.stamp = now.to_msg()
        self.position_pub.publish(self.setpoint)
        self.last_setpoint_time = now
        
        # Both leader and followers should apply separation rules
        if self.mission_state in ["TAKEOFF", "MISSION", "RETURN_HOME"]:
            # Apply separation vectors first to avoid collisions
            separation_vector = self.calculate_separation_vector()
            
            # Get current position
            my_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ])
            
            # Apply separation immediately for collision avoidance but only when truly needed
            # Reduced threshold to allow more movement since separation threshold is now smaller
            if np.linalg.norm(separation_vector) > 0.02:
                # Use PID controllers for smooth response
                self.setpoint.pose.position.x = my_pos[0] + separation_vector[0]
                self.setpoint.pose.position.y = my_pos[1] + separation_vector[1]
                self.setpoint.pose.position.z = my_pos[2] + separation_vector[2]
                
                # Log that separation is active
                self.get_logger().info(f"Separation active: vector={separation_vector}")
                
                # Don't apply other formation behaviors when separation is active
                return
            
            # If we're a follower, apply formation control after separation
            if not self.is_leader:
                self.apply_formation_control()

    def calculate_separation_vector(self):
        """Calculate separation vector to avoid collisions with other drones"""
        # Get current position
        my_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])
        
        # Initialize separation vector
        separation = np.zeros(3)
        
        # Check distance to all other drones
        for ns, info in self.other_drones.items():
            # Skip if data is too old (>1 second)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - info['last_update'] > 1.0:
                continue
                
            other_pos = np.array([
                info['pose'].pose.position.x,
                info['pose'].pose.position.y,
                info['pose'].pose.position.z
            ])
            
            # Vector from us to other drone
            diff = other_pos - my_pos
            distance = np.linalg.norm(diff)
            
            # Apply separation rule if too close
            if distance < self.separation_threshold and distance > 0:
                # Calculate repulsion force (inversely proportional to distance)
                # The closer the drones, the stronger the repulsion
                # Adjusted formula to be more aggressive at very close distances
                repulsion_strength = (self.separation_threshold / max(distance, 0.02)) - 1.0
                
                # Create normalized vector pointing away from the other drone
                repulsion_vector = -diff / distance * repulsion_strength * self.separation_weight
                
                # Add to separation vector
                separation += repulsion_vector
                
                # Log close proximity
                self.get_logger().info(
                    f"Close proximity detected with {ns}: distance={distance:.2f}m, " 
                    f"repulsion={np.linalg.norm(repulsion_vector):.2f}"
                )
        
        # Limit separation vector magnitude
        sep_mag = np.linalg.norm(separation)
        if sep_mag > 1.0:  # Increased max separation speed for faster response at close distances
            separation = separation / sep_mag * 1.0
            
        return separation

    def apply_formation_control(self):
        """Apply formation control with PID to maintain position relative to leader"""
        # Get current position
        my_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])
        
        # Find leader position
        leader_found = False
        for ns, info in self.other_drones.items():
            if "drone1" in ns:  # Assuming drone1 is the leader
                current_time = self.get_clock().now().nanoseconds / 1e9
                if current_time - info['last_update'] > 1.0:
                    continue  # Skip if data is too old
                    
                leader_pos = np.array([
                    info['pose'].pose.position.x,
                    info['pose'].pose.position.y,
                    info['pose'].pose.position.z
                ])
                leader_found = True
                
                # Calculate target position with offset
                target_pos = np.array([
                    leader_pos[0] + self.formation_offset['x'],
                    leader_pos[1] + self.formation_offset['y'],
                    leader_pos[2] + self.formation_offset['z']
                ])
                
                # Calculate distance to target
                distance_to_target = np.linalg.norm(target_pos - my_pos)
                
                # Use PID controllers for each axis
                vx = self.pid_x.update(target_pos[0], my_pos[0])
                vy = self.pid_y.update(target_pos[1], my_pos[1])
                vz = self.pid_z.update(target_pos[2], my_pos[2])
                
                # Scale time step based on distance for faster response when far away
                time_scale = min(0.25, max(0.1, distance_to_target / 10.0))
                
                # Update setpoint using PID outputs
                self.setpoint.pose.position.x = my_pos[0] + vx * time_scale
                self.setpoint.pose.position.y = my_pos[1] + vy * time_scale
                self.setpoint.pose.position.z = my_pos[2] + vz * time_scale
                
                # Log position data for debugging
                if distance_to_target > 1.0:
                    self.get_logger().info(
                        f"Following leader: distance={distance_to_target:.2f}m, "
                        f"target=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}), "
                        f"velocity=({vx:.2f}, {vy:.2f}, {vz:.2f})"
                    )
                
                break
                
        if not leader_found:
            # If leader not found, hover in place or proceed with backup plan
            self.get_logger().warn("Leader not found, maintaining position")

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
        # Reset PID controllers
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        
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
        self.setpoint.pose.position.x = self.current_pose.pose.position.x
        self.setpoint.pose.position.y = self.current_pose.pose.position.y
        self.setpoint.pose.position.z = self.current_pose.pose.position.z
    
    def calculate_distance(self, p1, p2):
        """Calculate 3D distance between two points"""
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (p1.z - p2.z) ** 2
        )
    
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
            
            # After 5 seconds, or when altitude is reached, transition to mission
            if elapsed > 5.0:
                self.mission_state = "MISSION"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, starting mission")
        
        elif self.mission_state == "MISSION":
            if self.is_leader:
                # Leader follows waypoints
                if self.waypoint_index < len(self.waypoints):
                    # Get current waypoint
                    wp = self.waypoints[self.waypoint_index]
                    
                    # Set the setpoint to the current waypoint
                    self.setpoint.pose.position.x = wp['x']
                    self.setpoint.pose.position.y = wp['y']
                    self.setpoint.pose.position.z = wp['z']
                    
                    # Check if we've reached the waypoint
                    current_pos = self.current_pose.pose.position
                    distance = self.calculate_distance(current_pos, Vector3(x=wp['x'], y=wp['y'], z=wp['z']))
                    
                    if distance < 0.5:  # Within 0.5m of the waypoint
                        self.waypoint_index += 1
                        self.get_logger().info(f"{self.namespace} Reached waypoint {self.waypoint_index}")
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
                # Followers just continue following the leader using formation control
                # This is handled in the publish_setpoint function
                
                # After a timeout, assume mission is complete
                if elapsed > 180.0:  # 3 minutes max mission time
                    self.mission_state = "RETURN_HOME"
                    self.state_start_time = current_time
                
        elif self.mission_state == "RETURN_HOME":
            # Set setpoint to home position
            self.setpoint.pose.position.x = self.home_position.pose.position.x
            self.setpoint.pose.position.y = self.home_position.pose.position.y
            self.setpoint.pose.position.z = self.home_position.pose.position.z
            
            # Check if we've reached home
            current_pos = self.current_pose.pose.position
            distance = self.calculate_distance(current_pos, self.home_position.pose.position)
            
            if distance < 0.5 or elapsed > 10.0:  # Within 0.5m of home or 10 seconds
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
    
    # Create drone instances - one leader and multiple followers
    leader = FormationDrone("drone1", is_leader=True)
    follower1 = FormationDrone("drone2", is_leader=False, formation_distance=3.0)
    follower2 = FormationDrone("drone3", is_leader=False, formation_distance=3.0)
    
    # Set formation offsets for followers
    follower1.set_formation_offset(x_offset=0.0, y_offset=3.0, z_offset=0.0)  # Follow 3m to the right
    follower2.set_formation_offset(x_offset=0.0, y_offset=-3.0, z_offset=0.0)  # Follow 3m to the left
    
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