#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleStatus, VehicleLocalPosition

class DroneStateListener(Node):
    def __init__(self): #MODIFY_NAME
        super().__init__('drone_state_listener')

        # Subscribers to PX4 topics
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, 10)

        # State variables
        self.armed = False
        self.flight_mode = None
        self.altitude = 0.0

    def vehicle_status_callback(self, msg):
        """Callback for vehicle status"""
        self.armed = msg.arming_state == 2  # 2 means ARMED
        flight_modes = ["MANUAL", "ALTCTL", "POSCTL", "AUTO_MISSION", "AUTO_LOITER", 
                        "AUTO_RTL", "AUTO_LAND", "AUTO_TAKEOFF", "OFFBOARD"]
        self.flight_mode = flight_modes[msg.nav_state] if msg.nav_state < len(flight_modes) else "UNKNOWN"

        self.get_logger().info(f"Armed: {self.armed} | Flight Mode: {self.flight_mode}")

def main(args=None):
    rclpy.init(args=args)
    node = DroneStateListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
