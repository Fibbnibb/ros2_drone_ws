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

class BoidsDrone(Node):
    def __init__(self, namespace, is_leader=False, formation_distance=2.0):
        super().__init__(f"drone_node_{namespace}")
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.is_leader = is_leader
        self.mission_completed = False
        self.formation_distance = formation_distance
        
        # Boids parameters
        self.separation_weight = 1.5  # Weight for separation rule
        self.alignment_weight = 1.0   # Weight for alignment rule
        self.cohesion_weight = 1.0    # Weight for cohesion rule
        self.leader_attraction_weight = 1.2  # Weight for leader attraction
        self.separation_threshold = 2.0  # Minimum distance between drones
        self.perception_radius = 5.0  # How far drones can "see" each other
        
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
        
        # If we're a follower, apply the boids algorithm to update the setpoint
        if not self.is_leader and self.mission_state in ["TAKEOFF", "MISSION", "RETURN_HOME"]:
            self.apply_boids_algorithm()

    def apply_boids_algorithm(self):
        """Apply the boids algorithm rules to update the setpoint"""
        # Get current position
        my_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])
        
        # Initialize vectors for each rule
        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        leader_attraction = np.zeros(3)
        
        # Find leader position
        leader_found = False
        for ns, info in self.other_drones.items():
            if 'leader' in ns:
                leader_pos = np.array([
                    info['pose'].pose.position.x,
                    info['pose'].pose.position.y,
                    info['pose'].pose.position.z
                ])
                leader_found = True
                
                # Calculate vector toward leader (will be used later)
                leader_vec = leader_pos - my_pos
                leader_dist = np.linalg.norm(leader_vec)
                
                # Only consider leader if within perception radius
                if leader_dist < self.perception_radius:
                    # Normalize and apply weight
                    if leader_dist > 0:
                        leader_attraction = leader_vec / leader_dist * self.leader_attraction_weight
                break
        
        # Count neighbors for averaging
        neighbors_count = 0
        sum_positions = np.zeros(3)  # For cohesion
        
        # Apply rules for each neighbor drone
        for ns, info in self.other_drones.items():
            # Skip if it's ourselves or the data is too old (>1 second)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - info['last_update'] > 1.0:
                continue
                
            other_pos = np.array([
                info['pose'].pose.position.x,
                info['pose'].pose.position.y,
                info['pose'].pose.position.z
            ])
            
            # Vector from us to neighbor
            diff = other_pos - my_pos
            distance = np.linalg.norm(diff)
            
            # Only consider drones within perception radius
            if distance < self.perception_radius and distance > 0:
                neighbors_count += 1
                
                # Rule 1: Separation - avoid getting too close
                if distance < self.separation_threshold:
                    # Vector pointing away from neighbor, with strength inversely proportional to distance
                    separation -= diff / distance * (self.separation_threshold / max(distance, 0.1)) * self.separation_weight
                
                # Sum positions for cohesion
                sum_positions += other_pos
        
        # Rule 3: Cohesion - move toward average position of neighbors
        if neighbors_count > 0:
            # Average position
            avg_position = sum_positions / neighbors_count
            # Vector toward average position
            cohesion = (avg_position - my_pos) * self.cohesion_weight
        
        # Calculate the final velocity vector by combining all rules
        velocity = separation + alignment + cohesion + leader_attraction
        
        # Normalize velocity if it's too large
        speed = np.linalg.norm(velocity)
        max_speed = 0.5  # Maximum allowed speed in m/s
        if speed > max_speed:
            velocity = velocity / speed * max_speed
        
        # Update setpoint
        # If leader was found, make sure we don't get too close or too far
        if leader_found:
            if leader_dist < self.formation_distance * 0.8:
                # Too close to leader, back off a bit
                self.setpoint.pose.position.x = my_pos[0] - leader_vec[0] * 0.1
                self.setpoint.pose.position.y = my_pos[1] - leader_vec[1] * 0.1
                self.setpoint.pose.position.z = my_pos[2]
            elif leader_dist > self.formation_distance * 1.2:
                # Too far from leader, get closer
                self.setpoint.pose.position.x = my_pos[0] + leader_vec[0] * 0.1
                self.setpoint.pose.position.y = my_pos[1] + leader_vec[1] * 0.1
                self.setpoint.pose.position.z = my_pos[2]
            else:
                # Within acceptable range, apply boids rules
                self.setpoint.pose.position.x = my_pos[0] + velocity[0]
                self.setpoint.pose.position.y = my_pos[1] + velocity[1]
                self.setpoint.pose.position.z = my_pos[2] + velocity[2]
        else:
            # No leader found, just apply boids rules
            self.setpoint.pose.position.x = my_pos[0] + velocity[0]
            self.setpoint.pose.position.y = my_pos[1] + velocity[1]
            self.setpoint.pose.position.z = my_pos[2] + velocity[2]
        
        # Maintain altitude
        self.setpoint.pose.position.z = 2.0

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
                # Followers just continue following the leader using boids rules
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
    leader = BoidsDrone("drone1", is_leader=True)
    follower1 = BoidsDrone("drone2", is_leader=False, formation_distance=3.0)
    follower2 = BoidsDrone("drone3", is_leader=False, formation_distance=3.0)
    #follower3 = BoidsDrone("drone_follower3", is_leader=False, formation_distance=3.0)
    
    # Register drones with each other
    leader.register_drone("drone2")
    leader.register_drone("drone3")
    #leader.register_drone("drone_follower3")
    
    follower1.register_drone("drone1")
    follower1.register_drone("drone3")
    #follower1.register_drone("drone_follower3")
    
    follower2.register_drone("drone1")
    follower2.register_drone("drone2")
    #follower2.register_drone("drone_follower3")
    
    # follower3.register_drone("drone_leader")
    # follower3.register_drone("drone_follower1")
    # follower3.register_drone("drone_follower2")
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=8)  # More threads for more drones
    executor.add_node(leader)
    executor.add_node(follower1)
    executor.add_node(follower2)
    #executor.add_node(follower3)
    
    try:
        # Start missions with a small delay between each
        time.sleep(1.0)
        leader.start_mission()
        time.sleep(1.0)
        follower1.start_mission()
        time.sleep(1.0)
        follower2.start_mission()
        #time.sleep(1.0)
        #follower3.start_mission()
        
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
                #and
                #follower3.mission_completed
            )
            
    except KeyboardInterrupt:
        # Return to home for all drones if interrupted
        leader.return_to_home()
        follower1.return_to_home()
        follower2.return_to_home()
        #follower3.return_to_home()
        
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
        #follower3.set_mode_async("AUTO.LAND")
        time.sleep(3.0)
        
    finally:
        # Cleanup
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()