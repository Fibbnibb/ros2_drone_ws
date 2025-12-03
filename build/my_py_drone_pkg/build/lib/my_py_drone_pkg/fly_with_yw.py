#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
import time
import math
import numpy as np

class PositionDrone(Node):
    def __init__(self, namespace="drone1"):
        super().__init__(f"position_drone_{namespace}")
        
        self.namespace = namespace
        self.current_state = State()
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.mission_state = "IDLE"
        
        # Simple target position you can change
        self.target_position = np.array([0.0, 0.0, 2.0])  # [x, y, z]
        
        # Control gains - tune these if needed
        self.kp_pos = 1.0    # Position gain
        self.kd_vel = 0.5    # Velocity damping
        
        # Create callback groups
        self.timer_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()
        
        # QoS Profile for sensor data
        # This defines the Quality of Service settings for subscriptions
        # best effort means messages are sent without guaranteed delivery
        # keep last history means only the most recent messages are kept in the queue
        # depth of 10 means the queue can hold up to 10 messages
        # This is suitable for high-frequency sensor data
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
            self.pose_callback, qos_profile)
        self.drone_vel_sub = self.create_subscription(
            TwistStamped, f'/{namespace}/local_position/velocity_local',
            self.velocity_callback, qos_profile)
        
        # Create service clients
        self.arm_client = self.create_client(
            CommandBool,
            f'/{namespace}/cmd/arming',
            callback_group=self.service_group
        )
        self.mode_client = self.create_client(
            SetMode,
            f'/{namespace}/set_mode',
            callback_group=self.service_group
        )
        
        # Create publishers
        self.attitude_pub = self.create_publisher(
            AttitudeTarget, f'/{namespace}/setpoint_raw/attitude', 10)
        self.shared_state_pub = self.create_publisher(
            TwistStamped, f'/shared_states/{namespace}', 10)
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for services...")
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            pass
            
        # Start control loop
        self.create_timer(0.05, self.control_loop, callback_group=self.timer_group)  # 20Hz
        self.create_timer(1.0, self.mission_step, callback_group=self.timer_group)   # 1Hz
        
        self.get_logger().info("Position drone initialized")

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        # Store current position
        self.position[0] = msg.pose.position.x
        self.position[1] = msg.pose.position.y
        self.position[2] = msg.pose.position.z

    def velocity_callback(self, msg):
        # Store current velocity from TwistStamped
        self.velocity[0] = msg.twist.linear.x
        self.velocity[1] = msg.twist.linear.y
        self.velocity[2] = msg.twist.linear.z

    def pos_to_attitude(self, current_pos, current_vel, target_pos):
        """Convert target position to attitude + thrust commands"""
        
        # 1) Position error
        pos_error = target_pos - current_pos
        vel_error = -current_vel  # We want zero velocity at target
        
        # 2) Desired acceleration (world frame)
        desired_accel = self.kp_pos * pos_error + self.kd_vel * vel_error
        
        # 3) Add gravity compensation
        desired_accel_world = desired_accel + np.array([0, 0, 9.81])
        
        # 4) Calculate thrust (total force magnitude)
        thrust = np.linalg.norm(desired_accel_world) / (9.81 * 2.0)  # Normalized
        thrust = max(0.0, min(1.0, thrust))  # Clamp to [0,1]
        
        # 5) Calculate attitude to point forward toward target
        horizontal_error = pos_error[:2]  # [x, y] only
        
        if np.linalg.norm(horizontal_error) > 0.1:  # If we need to move horizontally
            # Point forward toward the target
            yaw = math.atan2(horizontal_error[1], horizontal_error[0])
            
            # Small pitch forward to move faster (optional)
            pitch = -math.atan2(pos_error[2], np.linalg.norm(horizontal_error)) * 0.3
            pitch = max(-0.5, min(0.5, pitch))  # Limit pitch
        else:
            # Stay level if we're close to target
            yaw = 0.0
            pitch = 0.0
        
        roll = 0.0  # Keep level
        
        # 6) Convert to quaternion
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
        
        return [x, y, z, w], thrust

    def control_loop(self):
        """Main control loop - converts position target to attitude commands"""
        
        # Get attitude and thrust commands
        quat, thrust = self.pos_to_attitude(self.position, self.velocity, self.target_position)
        
        # Create attitude message
        att = AttitudeTarget()
        att.header.stamp = self.get_clock().now().to_msg()
        
        # Only control orientation and thrust, not body rates
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE | 
                        AttitudeTarget.IGNORE_PITCH_RATE | 
                        AttitudeTarget.IGNORE_YAW_RATE)
        
        # Set orientation
        att.orientation.x = quat[0]
        att.orientation.y = quat[1] 
        att.orientation.z = quat[2]
        att.orientation.w = quat[3]
        
        # Set thrust
        att.thrust = thrust
        
        self.attitude_pub.publish(att)
        
        # Publish shared state for other drones (for boid algorithms)
        shared_state = TwistStamped()
        shared_state.header.stamp = self.get_clock().now().to_msg()
        shared_state.header.frame_id = self.namespace
        shared_state.twist.linear.x = self.position[0]
        shared_state.twist.linear.y = self.position[1]
        shared_state.twist.linear.z = self.position[2]
        shared_state.twist.angular.x = self.velocity[0]
        shared_state.twist.angular.y = self.velocity[1]
        shared_state.twist.angular.z = self.velocity[2]
        self.shared_state_pub.publish(shared_state)

    def goto_position(self, x, y, z):
        """Tell drone to go to a specific position - CALL THIS FROM YOUR BOID CODE"""
        self.target_position = np.array([x, y, z])
        distance = np.linalg.norm(self.target_position - self.position)
        self.get_logger().info(f"New target: [{x:.2f}, {y:.2f}, {z:.2f}] (distance: {distance:.2f}m)")

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        self.arm_client.call_async(req)

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.mode_client.call_async(req)

    def mission_step(self):
        """Simple mission with waypoint demo"""
        if self.mission_state == "IDLE":
            self.mission_state = "ARM"
            self.get_logger().info("Starting mission...")
            
        elif self.mission_state == "ARM":
            self.arm()
            if self.current_state.armed:
                self.mission_state = "OFFBOARD"
                
        elif self.mission_state == "OFFBOARD":
            self.set_mode("OFFBOARD")
            if self.current_state.mode == "OFFBOARD":
                self.mission_state = "DEMO"
                self.demo_start_time = time.time()
                self.get_logger().info("Demo started - flying to waypoints")
                
        elif self.mission_state == "DEMO":
            # Demo: fly to different positions
            elapsed = time.time() - self.demo_start_time
            
            if elapsed < 10:
                self.goto_position(0.0, 0.0, 2.0)    # Hover at start
            elif elapsed < 20:
                self.goto_position(3.0, 0.0, 2.0)    # Forward
            elif elapsed < 30:
                self.goto_position(3.0, 3.0, 3.0)    # Forward-right and up
            elif elapsed < 40:
                self.goto_position(0.0, 3.0, 2.0)    # Left
            elif elapsed < 50:
                self.goto_position(-2.0, 0.0, 1.5)   # Back and down
            elif elapsed < 60:
                self.goto_position(0.0, 0.0, 2.0)    # Return home
            else:
                self.mission_state = "LAND"
                
        elif self.mission_state == "LAND":
            self.set_mode("AUTO.LAND")
            self.get_logger().info("Landing...")

    def get_position(self):
        """Get current position (for boid algorithms)"""
        return self.position.copy()

    def get_distance_to_target(self):
        """Check how close we are to target"""
        return np.linalg.norm(self.target_position - self.position)

def main():
    rclpy.init()
    
    # Create drone
    drone = PositionDrone("drone1")
    
    print("\n=== SIMPLE POSITION FOLLOWING DRONE ===")
    print("The drone will:")
    print("1. Face toward its target position")
    print("2. Move smoothly to reach it")
    print("3. Use altitude changes as needed")
    print("")
    print("Call drone.goto_position(x, y, z) to control it")
    print("Call drone.get_position() to see where it is")
    print("=====================================\n")
    
    try:
        rclpy.spin(drone)
    except KeyboardInterrupt:
        print("Emergency landing...")
        drone.set_mode("AUTO.LAND")
        time.sleep(3)
    
    drone.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()