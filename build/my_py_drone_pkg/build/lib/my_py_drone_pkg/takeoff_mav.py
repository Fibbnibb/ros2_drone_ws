from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

class TakeoffNode(Node):
    def __init__(self):
        super().__init__('takeoff_node')
        self.pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
        self.timer = self.create_timer(0.1, self.publish_setpoint)
        self.target_alt = 2.0  # target altitude in meters

    def publish_setpoint(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = self.target_alt
        self.pub.publish(pose)
        self.get_logger().info(f'Publishing takeoff setpoint at {self.target_alt} m')

def main(args=None):
    rclpy.init(args=args)
    node = TakeoffNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 