#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State

class MavrosStateSubscriber(Node):
    def __init__(self, namespace):
        super().__init__(f"mavros_state_subscriber_{namespace}")

        self.namespace = namespace

        self.subscription = self.create_subscription(
            State,
            f'/{namespace}/state',
            self.state_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Mavros State Subscriber has been started.')

    def state_callback(self, msg: State):
        self.get_logger().info(f"Received MAVROS state: "
                               f"armed={msg.armed}, "
                               f"connected={msg.connected}, "
                               f"guided={msg.guided}, "
                               f"mode={msg.mode}")

def main(args=None):
    rclpy.init(args=args)
    node = MavrosStateSubscriber("drone2")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
