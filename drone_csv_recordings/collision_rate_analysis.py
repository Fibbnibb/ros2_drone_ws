import pandas as pd
import numpy as np
import os
from itertools import combinations

def calculate_collision_rate(csv_file, collision_threshold=0.1):
    """
    Calculate collision rate for a drone session.
    
    Parameters:
    csv_file (str): Path to CSV file
    collision_threshold (float): Distance threshold for collision (meters)
    
    Returns:
    dict: Collision statistics
    """
    
    # Load data
    data = pd.read_csv(csv_file)
    
    # Use only needed columns
    data = data[['relative_time', 'x', 'y', 'z', 'drone_id']].dropna()
    
    collisions = 0
    total_checks = 0
    
    # Check each timestamp
    for time in data['relative_time'].unique():
        # Get all drones at this timestamp
        drones_at_time = data[data['relative_time'] == time]
        
        # Check all drone pairs
        drone_list = drones_at_time.to_dict('records')
        
        for drone1, drone2 in combinations(drone_list, 2):
            # Calculate 3D distance
            distance = np.sqrt(
                (drone1['x'] - drone2['x'])**2 + 
                (drone1['y'] - drone2['y'])**2 + 
                (drone1['z'] - drone2['z'])**2
            )
            
            total_checks += 1
            
            # Check if collision
            if distance <= collision_threshold:
                collisions += 1
    
    # Calculate rate
    collision_rate = (collisions / total_checks * 100) if total_checks > 0 else 0
    
    return {
        'session_file': csv_file,
        'total_collisions': collisions,
        'total_checks': total_checks,
        'collision_rate_percent': collision_rate,
        'num_drones': data['drone_id'].nunique(),
        'duration_seconds': data['relative_time'].max() - data['relative_time'].min()
    }

def select_csv_file():
    """
    Show available CSV files in current directory and let user choose.
    
    Returns:
    str: Selected CSV file name
    """
    # Get all CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory!")
        return None
    
    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input(f"\nSelect file (1-{len(csv_files)}): "))
            if 1 <= choice <= len(csv_files):
                return csv_files[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")

# Main execution
if __name__ == "__main__":
    # Select CSV file
    csv_file = select_csv_file()
    
    if csv_file:
        print(f"\nAnalyzing: {csv_file}")
        
        # Calculate collision rate
        result = calculate_collision_rate(csv_file)
        
        # Display results
        print("\n" + "="*50)
        print("COLLISION ANALYSIS RESULTS")
        print("="*50)
        print(f"Session: {result['session_file']}")
        print(f"Collisions: {result['total_collisions']}")
        print(f"Collision Rate: {result['collision_rate_percent']:.4f}%")
        print(f"Drones: {result['num_drones']}")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
        print(f"Total checks: {result['total_checks']:,}")
    else:
        print("No file selected. Exiting.")