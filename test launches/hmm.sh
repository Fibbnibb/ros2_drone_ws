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
