#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, Thrust
from mavros_msgs.srv import CommandBool, SetMode
import math
import time
import numpy as np

class SimpleAttitudeControl(Node):
    def __init__(self):
        super().__init__('simple_attitude_control')
        
        # Attitude targets (in degrees)
        self.target_roll = 0.0
        self.target_pitch = 0.0  
        self.target_yaw = 0.0
        self.target_thrust = 0.6  # 0.0 to 1.0
        
        # Position targets
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        
        # Flight state
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.is_armed = False
        self.offboard_enabled = False
        
        # Mission state
        self.mission_state = "INIT"
        self.mission_start_time = 0.0
        self.state_start_time = 0.0
        self.start_position = np.zeros(3)
        
        # Mission parameters
        self.takeoff_altitude = 2.1  # 7 feet ≈ 2.1 meters
        self.forward_distance_1 = 4.0  # 4 meters
        self.forward_distance_2 = 3.0  # 3 meters
        
        # Control mode flag
        self.use_position_control = True  # Use position control for takeoff
        
        # Create publishers - USING DRONE1 NAMESPACE
        self.attitude_pub = self.create_publisher(
            PoseStamped, '/drone1/setpoint_attitude/attitude', 10)
        
        self.thrust_pub = self.create_publisher(
            Thrust, '/drone1/setpoint_attitude/thrust', 10)
        
        # Position setpoint publisher
        self.position_pub = self.create_publisher(
            PoseStamped, '/drone1/setpoint_position/local', 10)
        
        # Create subscribers - USING DRONE1 NAMESPACE
        self.state_sub = self.create_subscription(
            State, '/drone1/state', self.state_callback, 10)
        
        self.pose_sub = self.create_subscription(
            PoseStamped, '/drone1/local_position/pose', 
            self.pose_callback, 10)
        
        # Create service clients - USING DRONE1 NAMESPACE
        self.arm_client = self.create_client(CommandBool, '/drone1/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/drone1/set_mode')
        
        # Wait for services
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set mode service...')
        
        # Create timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        
        self.get_logger().info('✅ Simple attitude control initialized with /drone1/ namespace')
        self.get_logger().info('📍 Using POSITION SETPOINTS for takeoff')

    def state_callback(self, msg):
        """Update flight state"""
        self.current_state = msg
        self.is_armed = msg.armed
        self.offboard_enabled = (msg.mode == "OFFBOARD")

    def pose_callback(self, msg):
        """Update current pose"""
        self.current_pose = msg

    def get_current_position(self):
        """Get current position as numpy array"""
        return np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])

    def get_current_altitude(self):
        """Get current altitude"""
        return self.current_pose.pose.position.z

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion"""
        # Convert to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch) 
        yaw = math.radians(yaw)
        
        # Calculate quaternion
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
        
        return [x, y, z, w]

    def control_loop(self):
        """Main control loop - publishes attitude/thrust or position setpoints"""
        if self.use_position_control:
            # Use position control
            self.publish_position_setpoint()
        else:
            # Use attitude control
            self.publish_attitude_setpoint()
        
        # Run mission state machine
        self.mission_step()

    def publish_position_setpoint(self):
        """Publish position setpoint"""
        position_msg = PoseStamped()
        position_msg.header.stamp = self.get_clock().now().to_msg()
        position_msg.header.frame_id = "map"
        
        # Set target position
        position_msg.pose.position.x = self.target_x
        position_msg.pose.position.y = self.target_y
        position_msg.pose.position.z = self.target_z
        
        # Set target orientation
        quat = self.euler_to_quaternion(self.target_roll, self.target_pitch, self.target_yaw)
        position_msg.pose.orientation.x = quat[0]
        position_msg.pose.orientation.y = quat[1]
        position_msg.pose.orientation.z = quat[2]
        position_msg.pose.orientation.w = quat[3]
        
        # Publish position setpoint
        self.position_pub.publish(position_msg)

    def publish_attitude_setpoint(self):
        """Publish attitude setpoint (original method)"""
        # Create attitude message
        attitude_msg = PoseStamped()
        attitude_msg.header.stamp = self.get_clock().now().to_msg()
        attitude_msg.header.frame_id = "base_link"
        
        # Set position (not used for attitude control, but required)
        attitude_msg.pose.position.x = 0.0
        attitude_msg.pose.position.y = 0.0
        attitude_msg.pose.position.z = 0.0
        
        # Convert target attitude to quaternion
        quat = self.euler_to_quaternion(self.target_roll, self.target_pitch, self.target_yaw)
        attitude_msg.pose.orientation.x = quat[0]
        attitude_msg.pose.orientation.y = quat[1]
        attitude_msg.pose.orientation.z = quat[2]
        attitude_msg.pose.orientation.w = quat[3]
        
        # Publish attitude
        self.attitude_pub.publish(attitude_msg)
        
        # Create and publish thrust message
        thrust_msg = Thrust()
        thrust_msg.header.stamp = self.get_clock().now().to_msg()
        thrust_msg.thrust = float(self.target_thrust)
        self.thrust_pub.publish(thrust_msg)

    def set_attitude(self, roll=0.0, pitch=0.0, yaw=0.0, thrust=0.6):
        """Set target attitude and thrust"""
        self.target_roll = roll
        self.target_pitch = pitch
        self.target_yaw = yaw
        self.target_thrust = max(0.0, min(1.0, thrust))  # Clamp between 0 and 1

    def set_position(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        """Set target position and yaw"""
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw

    def arm_async(self):
        """Non-blocking arm request"""
        request = CommandBool.Request()
        request.value = True
        future = self.arm_client.call_async(request)
        future.add_done_callback(self.arm_callback)

    def arm_callback(self, future):
        """Callback for arm service"""
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('✅ Armed successfully')
            else:
                self.get_logger().warn('❌ Arming failed - retrying...')
                # Retry arming after a short delay
                self.timer_single = self.create_timer(1.0, self.retry_arm)
        except Exception as e:
            self.get_logger().error(f"Arm service failed: {e}")
            self.timer_single = self.create_timer(1.0, self.retry_arm)

    def retry_arm(self):
        """Retry arming after delay"""
        self.timer_single.destroy()  # Remove the single-use timer
        self.arm_async()

    def set_mode_async(self, mode):
        """Non-blocking mode change request"""
        request = SetMode.Request()
        request.custom_mode = mode
        future = self.mode_client.call_async(request)
        future.add_done_callback(lambda f: self.mode_callback(f, mode))

    def mode_callback(self, future, mode):
        """Callback for set_mode service"""
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info(f'✅ Mode {mode} set successfully')
            else:
                self.get_logger().warn(f'❌ Failed to set mode {mode} - retrying...')
                # Retry mode setting after a short delay
                self.timer_single = self.create_timer(1.0, lambda: self.retry_mode(mode))
        except Exception as e:
            self.get_logger().error(f"Set mode service failed: {e}")
            self.timer_single = self.create_timer(1.0, lambda: self.retry_mode(mode))

    def retry_mode(self, mode):
        """Retry setting mode after delay"""
        self.timer_single.destroy()  # Remove the single-use timer
        self.set_mode_async(mode)

    def mission_step(self):
        """Mission state machine"""
        current_time = time.time()
        
        if self.mission_state == "INIT":
            self.get_logger().info("🚀 Starting mission...")
            self.get_logger().info("📡 Starting setpoint publishing before arming...")
            self.mission_state = "PUBLISH_SETPOINTS"
            self.state_start_time = current_time
            # Store initial position immediately
            self.start_position = self.get_current_position()
            self.get_logger().info(f"📍 Initial position: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f}, {self.start_position[2]:.2f})")
            
        elif self.mission_state == "PUBLISH_SETPOINTS":
            # Publish setpoints for 2 seconds before arming
            self.set_position(self.start_position[0], self.start_position[1], self.start_position[2], 0.0)
            
            elapsed = current_time - self.state_start_time
            if int(elapsed * 4) % 4 == 0:  # Log every 0.25 seconds
                self.get_logger().info(f"📡 Publishing setpoints... ({elapsed:.1f}s)")
            
            if elapsed > 2.0:  # Publish setpoints for 2 seconds
                self.get_logger().info("🔧 Arming drone...")
                self.arm_async()
                self.mission_state = "ARMING"
                self.state_start_time = current_time
                
        elif self.mission_state == "ARMING":
            # Keep sending setpoints while waiting for arm
            self.set_position(self.start_position[0], self.start_position[1], self.start_position[2], 0.0)
            
            if self.is_armed:
                self.get_logger().info("🚁 Drone armed! Setting OFFBOARD mode...")
                self.set_mode_async("OFFBOARD")
                self.mission_state = "SET_OFFBOARD"
                self.state_start_time = current_time
            else:
                # Log arming progress every 2 seconds
                if int(current_time - self.state_start_time) % 2 == 0:
                    elapsed = current_time - self.state_start_time
                    self.get_logger().info(f"⏱️  Waiting for arming... ({elapsed:.0f}s)")
                
        elif self.mission_state == "SET_OFFBOARD":
            # Keep sending setpoints while waiting for offboard mode
            self.set_position(self.start_position[0], self.start_position[1], self.start_position[2], 0.0)
            
            if self.offboard_enabled:
                self.get_logger().info("🚀 OFFBOARD mode active! Starting takeoff to 7 feet...")
                self.get_logger().info("📍 Using POSITION SETPOINTS for takeoff")
                self.mission_state = "TAKEOFF"
                self.state_start_time = current_time
            else:
                # Log mode setting progress
                if int(current_time - self.state_start_time) % 2 == 0:
                    elapsed = current_time - self.state_start_time
                    self.get_logger().info(f"⏱️  Waiting for OFFBOARD mode... ({elapsed:.0f}s)")
                
        elif self.mission_state == "TAKEOFF":
            # POSITION SETPOINT TAKEOFF to 7 feet
            target_altitude = self.start_position[2] + self.takeoff_altitude
            self.set_position(self.start_position[0], self.start_position[1], target_altitude, 0.0)
            
            current_altitude = self.get_current_altitude()
            altitude_error = target_altitude - current_altitude
            
            # Log takeoff progress every 0.5 seconds
            if int(current_time * 2) % 2 == 0:
                self.get_logger().info(f"🚁 Takeoff: {current_altitude:.2f}m / {target_altitude:.1f}m (error: {altitude_error:.2f}m)")
            
            if altitude_error < 0.3:  # Within 30cm of target
                self.get_logger().info(f"✅ Takeoff complete at {current_altitude:.1f}m!")
                self.mission_state = "HOVER"
                self.state_start_time = current_time
                
        elif self.mission_state == "HOVER":
            # Hover at takeoff altitude for a few seconds to stabilize
            target_altitude = self.start_position[2] + self.takeoff_altitude
            self.set_position(self.start_position[0], self.start_position[1], target_altitude, 0.0)
            
            elapsed = current_time - self.state_start_time
            self.get_logger().info(f"🚁 Hovering at {self.get_current_altitude():.2f}m... ({elapsed:.1f}s)")
            
            if elapsed > 3.0:  # Hover for 3 seconds
                self.get_logger().info("➡️  Starting forward movement (4 meters)...")
                self.get_logger().info("🔄 Switching to ATTITUDE CONTROL for maneuvering")
                self.use_position_control = False  # Switch to attitude control
                self.mission_state = "FORWARD_1"
                self.state_start_time = current_time
                
        elif self.mission_state == "FORWARD_1":
            # Go forward 4 meters using pitch (attitude control)
            thrust = self.calculate_altitude_thrust(self.start_position[2] + self.takeoff_altitude)
            self.set_attitude(roll=0, pitch=15, yaw=0, thrust=thrust)
            
            current_pos = self.get_current_position()
            distance_traveled = np.linalg.norm(current_pos[:2] - self.start_position[:2])
            
            # Log progress every 0.5 seconds
            if int(current_time * 2) % 2 == 0:
                self.get_logger().info(f"📐 Forward progress: {distance_traveled:.2f}m / {self.forward_distance_1:.1f}m")
            
            if distance_traveled >= self.forward_distance_1:
                self.get_logger().info(f"✅ Forward leg 1 complete! Traveled {distance_traveled:.1f}m")
                self.mission_state = "TURN"
                self.state_start_time = current_time
                
        elif self.mission_state == "TURN":
            # Turn using yaw and roll for 3 seconds
            elapsed = current_time - self.state_start_time
            thrust = self.calculate_altitude_thrust(self.start_position[2] + self.takeoff_altitude)
            
            if elapsed < 3.0:
                # Banking turn
                turn_progress = elapsed / 3.0
                turn_roll = 25.0  # Bank angle
                turn_yaw = 90.0 * turn_progress  # Turn 90 degrees total
                self.set_attitude(roll=turn_roll, pitch=0, yaw=turn_yaw, thrust=thrust)
                
                # Log turn progress
                self.get_logger().info(f"🔄 Turning: {turn_yaw:.1f}° (roll: {turn_roll:.1f}°)")
            else:
                self.get_logger().info("✅ Turn complete!")
                self.start_position = self.get_current_position()  # New reference
                self.mission_state = "FORWARD_2"
                self.state_start_time = current_time
                
        elif self.mission_state == "FORWARD_2":
            # Go forward 3 meters after turn
            thrust = self.calculate_altitude_thrust(self.start_position[2] + self.takeoff_altitude)
            self.set_attitude(roll=0, pitch=15, yaw=90, thrust=thrust)
            
            current_pos = self.get_current_position()
            distance_traveled = np.linalg.norm(current_pos[:2] - self.start_position[:2])
            
            # Log progress every 0.5 seconds
            if int(current_time * 2) % 2 == 0:
                self.get_logger().info(f"📐 Forward progress: {distance_traveled:.2f}m / {self.forward_distance_2:.1f}m")
            
            if distance_traveled >= self.forward_distance_2:
                self.get_logger().info(f"✅ Forward leg 2 complete! Traveled {distance_traveled:.1f}m")
                self.mission_state = "RETURN"
                self.state_start_time = current_time
                
        elif self.mission_state == "RETURN":
            # Return to hover
            elapsed = current_time - self.state_start_time
            thrust = self.calculate_altitude_thrust(self.start_position[2] + self.takeoff_altitude)
            
            if elapsed < 3.0:
                # Level out and hover
                self.set_attitude(roll=0, pitch=0, yaw=90, thrust=thrust)
                self.get_logger().info(f"🏠 Returning to hover... ({elapsed:.1f}s)")
            else:
                self.get_logger().info("🏠 Mission complete! Starting landing...")
                self.get_logger().info("📍 Switching back to POSITION CONTROL for landing")
                self.use_position_control = True  # Switch back to position control
                self.mission_state = "LAND"
                self.state_start_time = current_time
                
        elif self.mission_state == "LAND":
            # Position-controlled landing
            current_pos = self.get_current_position()
            elapsed = current_time - self.state_start_time
            
            # Gradual descent
            landing_altitude = max(self.start_position[2] + 0.1, 
                                 self.start_position[2] + self.takeoff_altitude - elapsed * 0.3)
            
            self.set_position(current_pos[0], current_pos[1], landing_altitude, 90.0)
            
            current_altitude = self.get_current_altitude()
            self.get_logger().info(f"🛬 Landing: {current_altitude:.2f}m (target: {landing_altitude:.2f}m)")
            
            if current_altitude < self.start_position[2] + 0.3 or elapsed > 25.0:
                self.get_logger().info("✅ Landing complete! Mission finished!")
                self.mission_state = "COMPLETE"
                
        elif self.mission_state == "COMPLETE":
            # Mission finished - land at starting position
            self.set_position(self.start_position[0], self.start_position[1], self.start_position[2], 0.0)

    def calculate_altitude_thrust(self, target_altitude):
        """Calculate thrust to maintain target altitude (for attitude control)"""
        current_altitude = self.get_current_altitude()
        altitude_error = target_altitude - current_altitude
        base_thrust = 0.6
        thrust_correction = altitude_error * 0.3  # P controller
        return max(0.4, min(0.8, base_thrust + thrust_correction))

def main():
    rclpy.init()
    
    try:
        print("\n================================================================")
        print("🚁 DRONE1 HYBRID CONTROL MISSION - Position + Attitude")
        print("================================================================")
        print("📋 Mission Plan:")
        print("  1. 📡 Publish setpoints (2 seconds)")
        print("  2. 🔧 Arm drone")
        print("  3. 🚀 Takeoff to 7 feet (2.1m) - POSITION SETPOINTS")
        print("  4. 🚁 Hover and stabilize - POSITION SETPOINTS")
        print("  5. ➡️  Go forward 4 meters - ATTITUDE CONTROL")
        print("  6. 🔄 Turn using yaw and roll - ATTITUDE CONTROL")
        print("  7. ➡️  Go forward 3 meters - ATTITUDE CONTROL") 
        print("  8. 🏠 Return and land - POSITION SETPOINTS")
        print("================================================================")
        print("📡 Using /drone1/ namespace:")
        print("   - /drone1/setpoint_position/local (takeoff/land)")
        print("   - /drone1/setpoint_attitude/attitude (maneuvers)")
        print("   - /drone1/setpoint_attitude/thrust (maneuvers)")
        print("   - /drone1/state")
        print("   - /drone1/local_position/pose")
        print("================================================================")
        
        # Create the attitude control node
        attitude_control = SimpleAttitudeControl()
        
        print("\n🚀 Mission starting...")
        print("📊 Watch the logs for detailed mission progress")
        print("📡 Will publish setpoints for 2 seconds before arming")
        print("📍 Takeoff will use POSITION SETPOINTS for stability")
        print("🔄 Maneuvers will use ATTITUDE CONTROL for agility")
        
        # Run the mission
        rclpy.spin(attitude_control)
        
    except KeyboardInterrupt:
        print("\n🚨 Mission interrupted by user")
        
        # Try to land safely
        if 'attitude_control' in locals():
            print("🛬 Emergency landing...")
            current_pos = attitude_control.get_current_position()
            attitude_control.use_position_control = True
            attitude_control.set_position(current_pos[0], current_pos[1], current_pos[2] - 0.5, 0.0)
            for i in range(40):  # 2 seconds of controlled descent
                rclpy.spin_once(attitude_control, timeout_sec=0.05)
        
    finally:
        print("🔌 Shutting down...")
        if 'attitude_control' in locals():
            attitude_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()