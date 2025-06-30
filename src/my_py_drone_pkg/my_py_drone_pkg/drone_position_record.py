#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import pandas as pd
import os
import signal
import sys
from datetime import datetime
import time
import glob
import re

class DronePositionRecorder(Node):
    def __init__(self):
        super().__init__('drone_position_recorder')
        
        # Output directory
        self.output_dir = '/home/emeka/drone_csv_recordings'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set recording frequency (5 Hz = every 0.2 seconds)
        self.recording_interval = 0.2  # seconds
        
        # Define drones to record
        self.drone_ids = ['drone1', 'drone2', 'drone3']
        
        # Determine flight number (increment from existing flights)
        self.flight_number = self.get_next_flight_number()
        self.flight_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.flight_name = f"Flight {self.flight_number} {self.flight_timestamp}"
        
        # Create data storage for each drone
        self.position_data = {}
        self.latest_position = {}
        
        for drone_id in self.drone_ids:
            self.position_data[drone_id] = {
                'timestamp': [],
                'x': [],
                'y': [],
                'z': [],
                'qx': [],
                'qy': [],
                'qz': [],
                'qw': []
            }
            
            self.latest_position[drone_id] = {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'qx': 0.0,
                'qy': 0.0,
                'qz': 0.0,
                'qw': 0.0,
                'updated': False
            }
        
        # Record start time
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Create custom QoS profile that's compatible with the publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.get_logger().info("Using BEST_EFFORT reliability QoS to match publisher")
        
        # Create subscriptions for each drone
        self.drone_subscriptions = []
        for drone_id in self.drone_ids:
            sub = self.create_subscription(
                PoseStamped,
                f'/{drone_id}/local_position/pose',
                lambda msg, id=drone_id: self.position_callback(msg, id),
                qos
            )
            self.drone_subscriptions.append(sub)
        
        # Create timer for recording at a fixed interval
        self.record_timer = self.create_timer(self.recording_interval, self.record_positions)
        
        # Status update timer
        self.status_timer = self.create_timer(5.0, self.log_status)
        
        # Setup clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.get_logger().info(f"Drone position recorder started: {self.flight_name}")
        self.get_logger().info(f"Recording position data for: {', '.join(self.drone_ids)}")
        self.get_logger().info(f"Recording frequency: {1.0/self.recording_interval:.1f} Hz (every {self.recording_interval} seconds)")
        self.get_logger().info(f"Data will be saved to: {self.output_dir}")
        self.get_logger().info("Press Ctrl+C to save and exit")
    
    def get_next_flight_number(self):
        """Find the next flight number by looking at existing files"""
        files = glob.glob(f"{self.output_dir}/Flight */*.csv")
        
        # Extract flight numbers from filenames
        flight_numbers = []
        pattern = r"Flight (\d+)"
        
        for file in files:
            match = re.search(pattern, file)
            if match:
                flight_number = int(match.group(1))
                flight_numbers.append(flight_number)
        
        # Find the next flight number
        if flight_numbers:
            return max(flight_numbers) + 1
        else:
            return 1
    
    def position_callback(self, msg, drone_id):
        """Store the latest position data for a specific drone"""
        # Update the latest position
        self.latest_position[drone_id]['x'] = msg.pose.position.x
        self.latest_position[drone_id]['y'] = msg.pose.position.y
        self.latest_position[drone_id]['z'] = msg.pose.position.z
        self.latest_position[drone_id]['qx'] = msg.pose.orientation.x
        self.latest_position[drone_id]['qy'] = msg.pose.orientation.y
        self.latest_position[drone_id]['qz'] = msg.pose.orientation.z
        self.latest_position[drone_id]['qw'] = msg.pose.orientation.w
        self.latest_position[drone_id]['updated'] = True
        
        # Log first message received for each drone
        if not self.position_data[drone_id]['timestamp'] and self.latest_position[drone_id]['updated']:
            self.get_logger().info(f"First position data received for {drone_id}!")
            self.get_logger().info(f"{drone_id} position: [{msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}]")
    
    def record_positions(self):
        """Record positions for all drones at a fixed interval"""
        # Get the current time
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Record data for each drone
        for drone_id in self.drone_ids:
            # Only record if this drone has received data
            if not self.latest_position[drone_id]['updated']:
                continue
                
            # Store the position data
            self.position_data[drone_id]['timestamp'].append(current_time)
            self.position_data[drone_id]['x'].append(self.latest_position[drone_id]['x'])
            self.position_data[drone_id]['y'].append(self.latest_position[drone_id]['y'])
            self.position_data[drone_id]['z'].append(self.latest_position[drone_id]['z'])
            self.position_data[drone_id]['qx'].append(self.latest_position[drone_id]['qx'])
            self.position_data[drone_id]['qy'].append(self.latest_position[drone_id]['qy'])
            self.position_data[drone_id]['qz'].append(self.latest_position[drone_id]['qz'])
            self.position_data[drone_id]['qw'].append(self.latest_position[drone_id]['qw'])
    
    def log_status(self):
        """Log recording status"""
        status_message = []
        
        for drone_id in self.drone_ids:
            count = len(self.position_data[drone_id]['timestamp'])
            if count > 0:
                status_message.append(f"{drone_id}: {count} points")
                
                # Show latest position periodically
                self.get_logger().info(f"{drone_id} latest position: [{self.latest_position[drone_id]['x']:.2f}, "
                                     f"{self.latest_position[drone_id]['y']:.2f}, {self.latest_position[drone_id]['z']:.2f}]")
        
        if status_message:
            self.get_logger().info(f"Recording in progress: {', '.join(status_message)}")
        else:
            self.get_logger().info("Waiting for position data...")
    
    def save_to_csv(self):
        """Save data to CSV files"""
        # Create flight directory if it doesn't exist
        flight_dir = f"{self.output_dir}/{self.flight_name}"
        os.makedirs(flight_dir, exist_ok=True)
        
        any_data_saved = False
        combined_data = []
        
        # Save individual files for each drone
        for drone_id in self.drone_ids:
            # Skip if no data for this drone
            if len(self.position_data[drone_id]['timestamp']) == 0:
                self.get_logger().warning(f"No position data was recorded for {drone_id}")
                continue
            
            # Create dataframe
            df = pd.DataFrame(self.position_data[drone_id])
            
            # Add relative time column
            df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
            
            # Save to file
            filename = f"{flight_dir}/{drone_id}.csv"
            df.to_csv(filename, index=False)
            
            # Calculate actual recording rate
            total_time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            actual_rate = len(df) / total_time if total_time > 0 else 0
            
            self.get_logger().info(f"Saved {len(df)} records for {drone_id} to {filename}")
            self.get_logger().info(f"{drone_id} recording rate: {actual_rate:.2f} Hz")
            
            # Add to combined data
            df['drone_id'] = drone_id
            combined_data.append(df)
            any_data_saved = True
        
        # Save combined file if we have data
        if any_data_saved and combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_filename = f"{flight_dir}/all_drones.csv"
            combined_df.to_csv(combined_filename, index=False)
            
            # Create a metadata file with flight information
            self.save_metadata(flight_dir, combined_df)
            
            self.get_logger().info(f"Saved combined data with {len(combined_df)} records to {combined_filename}")
        elif not any_data_saved:
            self.get_logger().warning("No position data was recorded for any drone")
    
    def save_metadata(self, flight_dir, df):
        """Save metadata about the flight"""
        metadata = {
            'flight_name': self.flight_name,
            'flight_number': self.flight_number,
            'timestamp': self.flight_timestamp,
            'drones': self.drone_ids,
            'recording_start': datetime.fromtimestamp(df['timestamp'].min()).strftime("%Y-%m-%d %H:%M:%S"),
            'recording_end': datetime.fromtimestamp(df['timestamp'].max()).strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min(),
            'total_points': len(df),
            'points_per_drone': {drone_id: len(df[df['drone_id'] == drone_id]) for drone_id in self.drone_ids if len(df[df['drone_id'] == drone_id]) > 0},
            'recording_frequency': 1.0 / self.recording_interval
        }
        
        # Write metadata as CSV
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(f"{flight_dir}/metadata.csv", index=False)
        
        # Also write as text for human readability
        with open(f"{flight_dir}/metadata.txt", 'w') as f:
            f.write(f"Flight Name: {metadata['flight_name']}\n")
            f.write(f"Flight Number: {metadata['flight_number']}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Drones: {', '.join(metadata['drones'])}\n")
            f.write(f"Recording Start: {metadata['recording_start']}\n")
            f.write(f"Recording End: {metadata['recording_end']}\n")
            f.write(f"Duration: {metadata['duration_seconds']:.2f} seconds\n")
            f.write(f"Total Data Points: {metadata['total_points']}\n")
            f.write("Points Per Drone:\n")
            for drone_id, count in metadata['points_per_drone'].items():
                f.write(f"  - {drone_id}: {count}\n")
            f.write(f"Recording Frequency: {metadata['recording_frequency']:.1f} Hz\n")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signal"""
        self.get_logger().info("Shutdown requested")
        self.save_to_csv()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = DronePositionRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save data when exiting
        node.save_to_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()