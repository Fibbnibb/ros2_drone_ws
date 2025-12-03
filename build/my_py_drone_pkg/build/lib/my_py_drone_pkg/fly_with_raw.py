import rclpy
from rclpy.node import Node
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Time

class RawVelocityCommander(Node):
    def __init__(self):
        super().__init__('raw_velocity_commander')
        self.publisher = self.create_publisher(PositionTarget, '/drone1/setpoint_raw/local', 10)
        self.timer = self.create_timer(0.1, self.send_velocity_command)  # 10 Hz

    def send_velocity_command(self):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # NED = North-East-Down

        # Set only VELOCITY + YAW_RATE
        msg.type_mask = (
            PositionTarget.IGNORE_PX |  # Ignore position
            PositionTarget.IGNORE_PY |
            PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AX |  # Ignore acceleration
            PositionTarget.IGNORE_AY |
            PositionTarget.IGNORE_AZ |
            PositionTarget.IGNORE_YAW   # <-- only using yaw_rate
        )

        # Velocity: move forward (x = 1 m/s)
        msg.velocity.x = 1.0  # forward
        msg.velocity.y = 0.0
        msg.velocity.z = 0.0  # maintain altitude

        # Yaw rate: rotate clockwise at 0.3 rad/s
        msg.yaw_rate = 0.3

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RawVelocityCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
