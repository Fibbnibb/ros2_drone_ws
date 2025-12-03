#!/bin/bash
# Define paths (Ensure NO spaces around `=`)
PX4_DIR="/home/emeka/PX4-Autopilot"  # Adjust if needed
QGD_DIR="/home/emeka/Downloads"
source /opt/ros/humble/setup.bash

# Make sure xdotool is installed
command -v xdotool >/dev/null 2>&1 || { echo "xdotool is required but not installed. Install with: sudo apt install xdotool"; exit 1; }

# Kill any existing processes
#killall gz PX4 QGroundControl.AppImage 2>/dev/null

# Launch QGroundControl
#gnome-terminal --title="QGroundControl" -- bash -c "cd ${QGD_DIR} && ./QGroundControl.AppImage; exec bash"
#(sleep 5 && xdotool search --name "QGroundControl" windowminimize) &

sleep 5

# Start Drone 0 (PX4 Instance 1) with x500 model - headless mode
gnome-terminal --title="PX4-DRONE1" -- bash -c "cd ${PX4_DIR} && \
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=x500 \
PX4_GZ_WORLD=lawn \
PX4_GZ_MODEL_POSE='0,0,0' \
MAV_SYS_ID=1 \
SITL_UDP_PRT=14580 \
PX4_GZ_GUI=1 \
./build/px4_sitl_default/bin/px4 -i 0; \
exec bash"
(sleep 5 && xdotool search --name "PX4-DRONE1" windowminimize) &

# Wait for PX4 to initialize
sleep 7

# Start Drone 1 (PX4 Instance 2) at different position - headless mode
gnome-terminal --title="PX4-DRONE2" -- bash -c "cd ${PX4_DIR} && \
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=x500 \
PX4_GZ_WORLD=lawn \
PX4_GZ_MODEL_POSE='0,1,0' \
MAV_SYS_ID=2 \
SITL_UDP_PRT=14581 \
PX4_GZ_GUI=0 \
./build/px4_sitl_default/bin/px4 -i 1; \
exec bash"
(sleep 5 && xdotool search --name "PX4-DRONE2" windowminimize) &

# Wait for PX4 to initialize
sleep 5

# Start Drone 3 (PX4 Instance 3) at different position - headless mode
gnome-terminal --title="PX4-DRONE3" -- bash -c "cd ${PX4_DIR} && \
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=x500 \
PX4_GZ_WORLD=lawn \
PX4_GZ_MODEL_POSE='0,-1,0' \
MAV_SYS_ID=3 \
SITL_UDP_PRT=14582 \
PX4_GZ_GUI=0 \
./build/px4_sitl_default/bin/px4 -i 2; \
exec bash"
(sleep 5 && xdotool search --name "PX4-DRONE3" windowminimize) &

# Wait for PX4 to initialize
sleep 5

# Start Drone 4 (PX4 Instance 3) at different position - headless mode
gnome-terminal --title="PX4-DRONE4" -- bash -c "cd ${PX4_DIR} && \
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=x500 \
PX4_GZ_WORLD=lawn \
PX4_GZ_MODEL_POSE='-1,0,0' \
MAV_SYS_ID=4 \
SITL_UDP_PRT=14583 \
PX4_GZ_GUI=0 \
./build/px4_sitl_default/bin/px4 -i 3; \
exec bash"
(sleep 5 && xdotool search --name "PX4-DRONE4" windowminimize) &

sleep 2

# Wait a bit before launching ROS
sleep 5

# Launch MAVROS nodes with minimization
gnome-terminal --title="MAVROS-DRONE1" -- bash -c "source /opt/ros/humble/setup.bash; ros2 run mavros mavros_node --ros-args -p fcu_url:=\"udp://:14540@127.0.0.1:14580\" -p system_id:=1 -p tgt_system:=1 --remap __ns:=/drone1; exec bash"
(sleep 5 && xdotool search --name "MAVROS-DRONE1" windowminimize) &

sleep 2

gnome-terminal --title="MAVROS-DRONE2" -- bash -c "source /opt/ros/humble/setup.bash; ros2 run mavros mavros_node --ros-args -p fcu_url:=\"udp://:14541@127.0.0.1:14581\" -p system_id:=2 -p tgt_system:=2 --remap __ns:=/drone2; exec bash"
(sleep 5 && xdotool search --name "MAVROS-DRONE2" windowminimize) &

sleep 2

gnome-terminal --title="MAVROS-DRONE3" -- bash -c "source /opt/ros/humble/setup.bash; ros2 run mavros mavros_node --ros-args -p fcu_url:=\"udp://:14542@127.0.0.1:14582\" -p system_id:=3 -p tgt_system:=3 --remap __ns:=/drone3; exec bash"
(sleep 5 && xdotool search --name "MAVROS-DRONE3" windowminimize) &

sleep 2

gnome-terminal --title="MAVROS-DRONE4" -- bash -c "source /opt/ros/humble/setup.bash; ros2 run mavros mavros_node --ros-args -p fcu_url:=\"udp://:14543@127.0.0.1:14583\" -p system_id:=4 -p tgt_system:=4 --remap __ns:=/drone4; exec bash"
(sleep 5 && xdotool search --name "MAVROS-DRONE4" windowminimize) &

sleep 5 &

gnome-terminal --title="BOID-NODE" -- bash -c "source /opt/ros/humble/setup.bash; source /home/emeka/ros2_drone_ws/install/setup.bash; 
ros2 run my_py_drone_pkg fly_4_drones; exec bash"
(sleep 5 && xdotool search --name "BOID-NODE" windowminimize)

sleep 2

gnome-terminal --title="POS-REC" -- bash -c "source /opt/ros/humble/setup.bash; source /home/emeka/ros2_drone_ws/install/setup.bash; 
ros2 run my_py_drone_pkg drone_pos_rec_3; exec bash"
(sleep 5 && xdotool search --name "POS-REC" windowminimize)  

# Print a helpful message
echo "All systems started! Windows will be minimized after 5 seconds each."
echo "- QGroundControl running"
echo "- PX4 SITL instances running for drone1, drone2, drone3, drone4"
echo "- MAVROS nodes running for all drones"
