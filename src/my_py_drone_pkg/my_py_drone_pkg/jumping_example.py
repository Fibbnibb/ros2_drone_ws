#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import time

class JumpingPersonNode(Node):
    def __init__(self, name, jump_interval):
        super().__init__(f'jumping_person_{name}')
        
        # Person-specific parameters
        self.name = name
        self.jump_interval = jump_interval
        self.jump_count = 0
        self.start_time = time.time()
        
        # Create a timer for jumping
        self.jump_timer = self.create_timer(
            jump_interval, 
            self.jump
        )
    
    def jump(self):
        # Check if 20 seconds have passed
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time < 20:
            # Increment jump count
            self.jump_count += 1
            
            # Log the jump with timestamp
            self.get_logger().info(
                f' jumped! '
                f'Jump #{self.jump_count} '
                f'at {current_time - self.start_time:.2f} seconds'
            )
        else:
            # Stop jumping after 20 seconds
            self.jump_timer.cancel()
            self.get_logger().info(
                f'{self.name} final jump count: {self.jump_count}'
            )

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create two jumping persons with different intervals
    person1 = JumpingPersonNode('Person1', 1.0)  # Jumps every 1 second
    person2 = JumpingPersonNode('Person2', 2.0)  # Jumps every 2 seconds
    
    # Create MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    
    # Add nodes to executor
    executor.add_node(person1)
    executor.add_node(person2)
    
    try:
        # Start the time tracking
        start_time = time.time()
        
        # Spin the executor for 20 seconds
        while time.time() - start_time < 6:
            executor.spin_once()
        
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()