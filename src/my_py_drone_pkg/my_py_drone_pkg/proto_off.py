import asyncio
import threading
import rclpy
from rclpy.node import Node
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)

class MyRosNode(Node):
    def __init__(self):
        super().__init__("my_ros_node")
        # Publishers, subscribers, timers, etc.
        self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("ROS timer tick...")

async def mavsdk_task():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break
    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break
    
    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return
    
    print("Arming via MAVSDK...")
    await drone.action.arm()
    print("Taking off...")
    await drone.action.takeoff()


    
    await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -5.0, 0.0)) # x, y, z, yaw
    print("-- Go 0m North, 0m East, -5m Down \
            within local coordinate system")
    await asyncio.sleep(10)
    # Continue MAVSDK operations, or just keep a loop going:
    while True:
        await asyncio.sleep(1)
    

def ros_spin_thread(ros_node):
    rclpy.spin(ros_node)

async def main_async():
    rclpy.init()
    node = MyRosNode()

    # Start ROS spin in a thread
    ros_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    ros_thread.start()

    # Meanwhile, run MAVSDK in this asyncio loop
    await mavsdk_task()

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
