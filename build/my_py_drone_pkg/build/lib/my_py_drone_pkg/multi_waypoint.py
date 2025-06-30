#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Quaternion
from my_drone_interfaces.msg import LeaderFollower, Neighbor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
from tf_transformations import quaternion_from_euler

class Drone(Node):
    def __init__(self, namespace, is_leader=False):
        super().__init__(f"drone_node_{namespace}")
        
        # Store namespace and mission parameters
        self.namespace = namespace
        self.mission_completed = False
        self.is_leader = is_leader
        self.neighbors = {}  # Dictionary to store neighbor drones and distances
        
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
        
        self.drone_height_sub = self.create_subscription(
            PoseStamped, f'/{namespace}/local_position/pose', 
            self.drone_height_callback, qos_profile)
        
        # Leader-follower subscriptions and publishers
        self.leader_follower_pub = self.create_publisher(
            LeaderFollower, '/leader_follower_info', 10)
            
        # Subscribe to all drones' positions to calculate distances
        self.drone_positions = {}
        self.drone1_pos_sub = self.create_subscription(
            PoseStamped, '/drone1/local_position/pose', 
            lambda msg: self.drone_position_callback(msg, 'drone1'), 
            qos_profile)
            
        self.drone2_pos_sub = self.create_subscription(
            PoseStamped, '/drone2/local_position/pose', 
            lambda msg: self.drone_position_callback(msg, 'drone2'), 
            qos_profile)
        
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
        
        # Initial orientation (pointing north)
        q = quaternion_from_euler(0, 0, 0)  # Roll, Pitch, Yaw in radians
        self.setpoint.pose.orientation.x = q[0]
        self.setpoint.pose.orientation.y = q[1]
        self.setpoint.pose.orientation.z = q[2]
        self.setpoint.pose.orientation.w = q[3]
        
        # Store home position
        self.home_position = PoseStamped()
        self.home_position.pose.position.x = 0.0
        self.home_position.pose.position.y = 0.0
        self.home_position.pose.position.z = 2.0
        self.home_position.pose.orientation = self.setpoint.pose.orientation
        
        # Mission state variables
        self.mission_state = "IDLE"
        self.target_distance = 0.0
        self.turn_direction = "LEFT"  # or "RIGHT"
        self.turn_angle = 0.0  # in radians
        self.target_yaw = 0.0  # Current target yaw in radians
        self.current_yaw = 0.0  # Current drone yaw in radians
        self.state_start_time = 0.0
        self.last_setpoint_time = self.get_clock().now()
        
        # Waypoints for multi-leg missions
        self.waypoints = []
        self.current_waypoint_index = 0
        
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
        
        # Leader-follower timer
        self.leader_follower_timer = self.create_timer(
            1.0,  # Update leader-follower info once per second
            self.publish_leader_follower_info,
            callback_group=self.timer_group
        )
        
        self.get_logger().info(f"{namespace} initialized as {'leader' if is_leader else 'follower'}")

    def set_yaw(self, yaw_radians):
        """Set the yaw of the drone"""
        q = quaternion_from_euler(0, 0, yaw_radians)
        self.setpoint.pose.orientation.x = q[0]
        self.setpoint.pose.orientation.y = q[1]
        self.setpoint.pose.orientation.z = q[2]
        self.setpoint.pose.orientation.w = q[3]
        self.target_yaw = yaw_radians

    def turn(self, direction, angle_degrees):
        """Turn the drone left or right by specified angle in degrees"""
        angle_radians = math.radians(angle_degrees)
        
        # Calculate new yaw based on direction
        if direction == "LEFT":
            new_yaw = self.target_yaw + angle_radians
        else:  # RIGHT
            new_yaw = self.target_yaw - angle_radians
            
        # Normalize yaw to [-pi, pi]
        new_yaw = ((new_yaw + math.pi) % (2 * math.pi)) - math.pi
        
        self.turn_direction = direction
        self.turn_angle = angle_radians
        self.target_yaw = new_yaw
        
        self.get_logger().info(f"{self.namespace} turning {direction} by {angle_degrees} degrees")
        return new_yaw

    def drone_position_callback(self, msg, drone_id):
        """Store position of each drone to calculate distances"""
        self.drone_positions[drone_id] = msg
        
        # Skip self
        if drone_id == self.namespace:
            return
            
        # Calculate distance if we have our own position
        if self.namespace in self.drone_positions:
            my_pos = self.drone_positions[self.namespace].pose.position
            other_pos = msg.pose.position
            
            # Calculate Euclidean distance
            distance = math.sqrt(
                (my_pos.x - other_pos.x) ** 2 +
                (my_pos.y - other_pos.y) ** 2 +
                (my_pos.z - other_pos.z) ** 2
            )
            
            # Update neighbor info
            self.neighbors[drone_id] = distance

    def publish_leader_follower_info(self):
        """Publish leader-follower information"""
        msg = LeaderFollower()
        msg.is_leader = self.is_leader
        msg.drone_id = self.namespace
        
        # Build list of neighbors
        neighbor_list = []
        for drone_id, distance in self.neighbors.items():
            neighbor = Neighbor()
            neighbor.drone_id = drone_id
            neighbor.distance = distance
            neighbor_list.append(neighbor)
            
        msg.neighbors = neighbor_list
        
        # Publish
        self.leader_follower_pub.publish(msg)

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
    
    def start_mission(self, waypoints=None):
        """Start the mission state machine with optional waypoints list
        
        waypoints should be a list of dicts with keys:
        - x, y, z: position coordinates
        - yaw: yaw in degrees (optional)
        - duration: how long to stay at this waypoint in seconds
        """
        self.get_logger().info(f"{self.namespace} Starting mission...")
        self.mission_state = "INIT"
        
        if waypoints:
            self.waypoints = waypoints
            self.current_waypoint_index = 0
            self.get_logger().info(f"{self.namespace} Mission with {len(waypoints)} waypoints")
        else:
            # Default single waypoint mission (forward 5m)
            self.waypoints = [
                {'x': 5.0, 'y': 0.0, 'z': 2.0, 'yaw': 0.0, 'duration': 5.0}
            ]
        
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
    
    def follow_leader(self):
        """Calculate and update setpoint to follow the leader"""
        # Find the leader in neighbors
        leader_id = None
        for drone_id, distance in self.neighbors.items():
            if drone_id == 'drone1' and self.namespace != 'drone1':
                leader_id = drone_id
                break
                
        if leader_id and leader_id in self.drone_positions:
            # Get leader position
            leader_pos = self.drone_positions[leader_id].pose.position
            leader_orient = self.drone_positions[leader_id].pose.orientation
            
            # Set setpoint to follow leader with an offset (e.g., 2m behind)
            if self.mission_state == "MOVE_TO_WAYPOINT" or self.mission_state == "FOLLOW_LEADER":
                self.setpoint.pose.position.x = leader_pos.x - 2.0
                self.setpoint.pose.position.y = leader_pos.y
                self.setpoint.pose.position.z = leader_pos.z
                
                # Match leader's orientation
                self.setpoint.pose.orientation = leader_orient
    
    def move_to_waypoint(self, waypoint_index):
        """Set setpoint to move to the specified waypoint"""
        if 0 <= waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[waypoint_index]
            
            # Set position
            self.setpoint.pose.position.x = waypoint['x']
            self.setpoint.pose.position.y = waypoint['y']
            self.setpoint.pose.position.z = waypoint['z']
            
            # Set orientation if specified
            if 'yaw' in waypoint:
                self.set_yaw(math.radians(waypoint['yaw']))
            
            return True
        return False
    
    def mission_step(self):
        """Mission state machine timer callback"""
        if self.mission_state == "IDLE":
            # Do nothing
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.state_start_time
        
        # If this is a follower and not a leader, it should follow the leader
        if not self.is_leader and (self.mission_state == "MOVE_TO_WAYPOINT" or 
                                 self.mission_state == "TURNING" or
                                 self.mission_state == "FOLLOW_LEADER"):
            self.mission_state = "FOLLOW_LEADER"
            self.follow_leader()
            return
            
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
            
            # After 5 seconds, or when altitude is reached, transition to waypoint navigation
            if elapsed > 5.0:
                if self.is_leader:
                    self.mission_state = "MOVE_TO_WAYPOINT"
                    self.move_to_waypoint(self.current_waypoint_index)
                else:
                    self.mission_state = "FOLLOW_LEADER"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Takeoff complete, " + 
                                     ("moving to waypoint" if self.is_leader else "following leader"))
        
        elif self.mission_state == "MOVE_TO_WAYPOINT":
            # Only the leader executes waypoint navigation directly
            if not self.is_leader:
                return
                
            # Check if we've been at this waypoint long enough
            current_waypoint = self.waypoints[self.current_waypoint_index]
            waypoint_duration = current_waypoint.get('duration', 5.0)
            
            if elapsed > waypoint_duration:
                # Move to next waypoint or finish
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index < len(self.waypoints):
                    # Move to next waypoint
                    self.move_to_waypoint(self.current_waypoint_index)
                    self.state_start_time = current_time
                    self.get_logger().info(f"{self.namespace} Moving to waypoint {self.current_waypoint_index+1}/{len(self.waypoints)}")
                else:
                    # Finished all waypoints, return home
                    self.mission_state = "RETURN_HOME"
                    self.state_start_time = current_time
                    self.get_logger().info(f"{self.namespace} All waypoints visited, returning home")
        
        elif self.mission_state == "TURNING":
            # Only the leader executes turning directly
            if not self.is_leader:
                return
                
            # Check if turn is complete (based on time)
            if elapsed > 3.0:  # Allow 3 seconds for turn to complete
                self.mission_state = "MOVE_TO_WAYPOINT"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Turn complete, resuming waypoint navigation")
        
        elif self.mission_state == "FOLLOW_LEADER":
            # The follower stays in this state until the leader changes state
            # Logic for following is in follow_leader()
            if elapsed > 60.0:  # Safety timeout
                self.mission_state = "RETURN_HOME"
                self.state_start_time = current_time
                self.get_logger().info(f"{self.namespace} Follow timeout, returning home")
                
        elif self.mission_state == "RETURN_HOME":
            # Gradually move back to home position (X=0, Y=0)
            # Calculate how far we've gone in return journey as a percentage (0-1)
            return_progress = min(elapsed / 10.0, 1.0)  # Complete return in 10 seconds
            
            # Linear interpolation between current position and home
            current_x = self.current_pose.pose.position.x
            current_y = self.current_pose.pose.position.y
            
            target_x = current_x * (1.0 - return_progress)
            target_y = current_y * (1.0 - return_progress)
            
            self.setpoint.pose.position.x = target_x
            self.setpoint.pose.position.y = target_y
            
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
    
    # Create drone instances - drone1 is leader, drone2 is follower
    drone1 = Drone("drone1", is_leader=True)
    drone2 = Drone("drone2", is_leader=False)
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(drone1)
    executor.add_node(drone2)
    
    try:
        # Define waypoints for leader with turns
        leader_waypoints = [
            # First leg - forward
            {'x': 5.0, 'y': 0.0, 'z': 2.0, 'yaw': 0.0, 'duration': 5.0},
            # Turn left (90 degrees)
            {'x': 5.0, 'y': 0.0, 'z': 2.0, 'yaw': 90.0, 'duration': 3.0},
            # Second leg - left
            {'x': 5.0, 'y': 5.0, 'z': 2.0, 'yaw': 90.0, 'duration': 5.0},
            # Turn right (90 degrees)
            {'x': 5.0, 'y': 5.0, 'z': 2.0, 'yaw': 0.0, 'duration': 3.0},
            # Third leg - forward again
            {'x': 10.0, 'y': 5.0, 'z': 2.0, 'yaw': 0.0, 'duration': 5.0}
        ]
        
        # Start missions
        drone1.start_mission(leader_waypoints)   # Leader follows waypoints
        drone2.start_mission()                   # Follower follows the leader
        
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