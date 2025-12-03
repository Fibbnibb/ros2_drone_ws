#!/usr/bin/env python3

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import glob
import numpy as np
import re
from datetime import datetime

# Configure Dash app with the correct settings
app = dash.Dash(__name__, 
                prevent_initial_callbacks='initial_duplicate')  # Fix duplicate callback issue

# Get list of available flights and CSV files
def get_available_flight_data():
    csv_dir = '/home/emeka/drone_csv_recordings'
    
    # List to store all flight options
    flight_options = []
    
    # 1. Check for flight directories
    flight_dirs = glob.glob(f"{csv_dir}/Flight *")
    for flight_dir in sorted(flight_dirs, reverse=True):
        flight_name = os.path.basename(flight_dir)
        
        # Check if the flight directory has a CSV file
        csv_files = glob.glob(f"{flight_dir}/*.csv")
        if csv_files:
            # Use the first CSV file that contains "all_drones" or just the first one if none match
            primary_csv = next((f for f in csv_files if "all_drones" in os.path.basename(f).lower()), csv_files[0])
            
            # Read metadata if available for description
            description = flight_name
            metadata_file = f"{flight_dir}/metadata.txt"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        first_lines = [next(f) for _ in range(5) if f]
                        # Extract duration if available
                        duration_line = next((line for line in first_lines if "Duration:" in line), None)
                        if duration_line:
                            try:
                                duration = float(re.search(r"Duration: ([\d.]+)", duration_line).group(1))
                                description = f"{flight_name} ({duration:.1f}s)"
                            except:
                                pass
                except:
                    pass
            
            flight_options.append({
                "label": f"📁 {description}",
                "value": primary_csv
            })
    
    # 2. Check for standalone CSV files in the main directory
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    for csv_file in sorted(csv_files, reverse=True):
        # Skip non-position CSV files
        filename = os.path.basename(csv_file)
        if "position" in filename.lower() or "drone" in filename.lower():
            # Try to extract a timestamp from the filename
            date_match = re.search(r'(\d{8}_\d{6})', filename)
            if date_match:
                date_str = date_match.group(1)
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    formatted_date = file_date.strftime("%Y-%m-%d %H:%M")
                    description = f"CSV: {formatted_date}"
                except:
                    description = f"CSV: {filename}"
            else:
                description = f"CSV: {filename}"
            
            flight_options.append({
                "label": description,
                "value": csv_file
            })
    
    if not flight_options:
        return [{"label": "No flight data available", "value": "none"}]
    
    return flight_options

# Load the drone position data from a CSV file
def load_flight_data(csv_file):
    if csv_file == "none" or not os.path.exists(csv_file):
        # Return empty dataframe with required columns if no file found
        return pd.DataFrame(columns=['timestamp', 'relative_time', 'x', 'y', 'z', 'drone_id'])
    
    # Load the data
    df = pd.DataFrame()
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data from {csv_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if the dataframe has the necessary columns
        required_cols = ['x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: CSV file missing required columns: {required_cols}")
            return pd.DataFrame(columns=['timestamp', 'relative_time', 'x', 'y', 'z', 'drone_id'])
        
        # If there's no drone_id column, try to determine from filename or add a default
        if 'drone_id' not in df.columns:
            filename = os.path.basename(csv_file)
            if "drone1" in filename:
                df['drone_id'] = 'drone1'
            elif "drone2" in filename:
                df['drone_id'] = 'drone2'
            elif "drone3" in filename:
                df['drone_id'] = 'drone3'
            else:
                df['drone_id'] = 'drone1'  # Default value
        
        # If there's no relative_time column, but we have timestamp, create it
        if 'relative_time' not in df.columns and 'timestamp' in df.columns:
            df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        # If we don't have either timestamp or relative_time, create a simple sequence
        if 'relative_time' not in df.columns:
            df['relative_time'] = [i * 0.2 for i in range(len(df))]  # Assume 5Hz data (0.2s between points)
            
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return df

# Load the most recent flight data by default
def get_default_flight_data():
    flight_options = get_available_flight_data()
    if flight_options and flight_options[0]["value"] != "none":
        return flight_options[0]["value"]
    return None

# Set the time increment for animation (0.5 seconds)
time_increment = 0.5

# Get available flight data
available_flights = get_available_flight_data()
default_flight = get_default_flight_data()

# Initial data load
df = load_flight_data(default_flight) if default_flight else pd.DataFrame(columns=['timestamp', 'relative_time', 'x', 'y', 'z', 'drone_id'])

# Calculate initial axis ranges
def calculate_axis_ranges(df):
    if not df.empty and 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        # Get the min and max for each axis with some padding (10%)
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        z_min, z_max = df['z'].min(), df['z'].max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # At least 5 meters range for visibility
        x_range = max(x_range, 5.0)
        y_range = max(y_range, 5.0)
        z_range = max(z_range, 5.0)
        
        padding = 0.1  # 10% padding
        x_min = x_min - x_range * padding
        x_max = x_max + x_range * padding
        y_min = y_min - y_range * padding
        y_max = y_max + y_range * padding
        z_min = z_min - z_range * padding
        z_max = z_max + z_range * padding
        
        # Make sure z has at least 0 as minimum for ground reference
        z_min = min(z_min, 0)
        
        # Get the max range to make cubic aspect ratio
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        # Set fixed ranges with cubic aspect ratio
        x_min, x_max = x_center - max_range/2, x_center + max_range/2
        y_min, y_max = y_center - max_range/2, y_center + max_range/2
        z_min, z_max = z_center - max_range/2, z_center + max_range/2
        
        return {
            'x': [x_min, x_max],
            'y': [y_min, y_max],
            'z': [z_min, z_max]
        }
    else:
        # Default values if no data
        return {
            'x': [-5, 5],
            'y': [-5, 5],
            'z': [0, 10]
        }

axis_ranges = calculate_axis_ranges(df)

# Prepare time slider values
def prepare_time_slider(df):
    if not df.empty and 'relative_time' in df.columns:
        time_marks = {}
        time_min = df['relative_time'].min()
        time_max = df['relative_time'].max()
        step = (time_max - time_min) / 10  # 10 marks on slider
        
        for i in range(11):
            value = time_min + i * step
            time_marks[value] = f"{value:.1f}s"
        
        return time_min, time_max, step, time_marks
    else:
        time_min, time_max, step = 0, 10, 1
        time_marks = {i: f"{i}s" for i in range(0, 11)}
        return time_min, time_max, step, time_marks

time_min, time_max, step, time_marks = prepare_time_slider(df)

# Define the app layout
app.layout = html.Div(style={'fontFamily': 'Arial, Helvetica, sans-serif'}, children=[
    html.H1("Drone Position Visualization", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label("Select Flight Data:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='flight-selector',
                options=available_flights,
                value=default_flight,
                clearable=False,
                style={'width': '50%', 'display': 'inline-block'}
            ),
            html.Button("Refresh Data List", 
                       id="refresh-button", 
                       n_clicks=0,
                       style={'marginLeft': '10px', 'padding': '8px 16px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'borderRadius': '4px'})
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        
        html.Div(id='flight-info', style={'textAlign': 'center', 'marginBottom': '10px', 'fontStyle': 'italic'}),
        
        html.Div([
            dcc.Graph(id='drone-position-plot', style={'height': '70vh'})
        ], style={'width': '100%', 'display': 'inline-block', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px'}),
        
        html.Div([
            html.Div([
                html.Label("Current Time:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Div(id='time-display', style={'fontSize': '24px', 'margin': '5px 0 15px 0', 'color': '#2c3e50'})
            ], style={'width': '100%', 'textAlign': 'center'}),
            
            dcc.Slider(
                id='time-slider',
                min=time_min,
                max=time_max,
                step=time_increment/2,  # Half of the time increment for finer control
                value=time_min,
                marks=time_marks,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Div([
                html.Button("Play", id="play-button", n_clicks=0, 
                           style={'marginRight': '10px', 'padding': '8px 16px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
                html.Button("Pause", id="pause-button", n_clicks=0,
                           style={'marginRight': '10px', 'padding': '8px 16px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
                html.Div(style={'display': 'inline-block', 'marginLeft': '20px'}, children=[
                    html.Label("Playback Speed:", style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='speed-dropdown',
                        options=[
                            {'label': 'Very Slow (0.25x)', 'value': 0.25},
                            {'label': 'Slow (0.5x)', 'value': 0.5},
                            {'label': 'Normal (1x)', 'value': 1.0},
                            {'label': 'Fast (2x)', 'value': 2.0}
                        ],
                        value=0.5,  # Default to slow speed
                        clearable=False,
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ])
            ], style={'margin': '20px 0', 'textAlign': 'center'}),
            
            html.Div([
                html.Label("Display Options:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Checklist(
                    id='display-options',
                    options=[
                        {'label': 'Show Trail Paths', 'value': 'trails'},
                        {'label': 'Show Reference Grid', 'value': 'grid'},
                        {'label': 'Show Drone Labels', 'value': 'labels'}
                    ],
                    value=['trails', 'grid', 'labels'],
                    inline=True
                )
            ], style={'margin': '10px 0', 'textAlign': 'center'}),
            
            dcc.Interval(
                id='animation-interval',
                interval=500,  # 500ms = 0.5 seconds by default
                n_intervals=0,
                disabled=True
            ),
            
            # Store for animation control state
            dcc.Store(id='animation-state')
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '20px', 'boxSizing': 'border-box'})
    ])
])

# Callback to update flight selector options when refresh button is clicked
@app.callback(
    Output('flight-selector', 'options'),
    [Input('refresh-button', 'n_clicks')]
)
def refresh_flight_list(n_clicks):
    return get_available_flight_data()

# Callback to update flight info when a flight is selected
@app.callback(
    Output('flight-info', 'children'),
    [Input('flight-selector', 'value')]
)
def update_flight_info(csv_file):
    if csv_file == "none" or not os.path.exists(csv_file):
        return "No flight data selected"
    
    # Check if this is from a flight directory and if metadata exists
    csv_dir = os.path.dirname(csv_file)
    metadata_file = f"{csv_dir}/metadata.txt"
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                first_lines = [next(f) for _ in range(5) if f]
                info = " | ".join([line.strip() for line in first_lines if line.strip()])
                return info
        except:
            pass
    
    # For standalone CSV files, just provide basic file info
    file_stats = os.stat(csv_file)
    file_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    file_size = file_stats.st_size / 1024  # Convert to KB
    
    # Try to get row count without loading the whole file
    row_count = 0
    try:
        with open(csv_file, 'r') as f:
            for _ in f:
                row_count += 1
        row_count -= 1  # Subtract header row
    except:
        df = load_flight_data(csv_file)
        row_count = len(df)
    
    return f"File: {os.path.basename(csv_file)} | Modified: {file_time} | Size: {file_size:.1f} KB | Rows: {row_count}"

# Callback to update the visualization when a flight is selected
@app.callback(
    [Output('time-slider', 'min'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'),
     Output('time-slider', 'value'),
     Output('animation-state', 'data')],
    [Input('flight-selector', 'value')]
)
def update_time_slider_range(csv_file):
    # Load the data for the selected flight
    df = load_flight_data(csv_file)
    
    # Prepare time slider values
    time_min, time_max, step, time_marks = prepare_time_slider(df)
    
    # Reset animation state
    animation_state = {"playing": False}
    
    # Reset slider to start
    return time_min, time_max, time_marks, time_min, animation_state

# Callback to update animation state based on play/pause buttons
@app.callback(
    [Output('animation-interval', 'disabled'),
     Output('animation-state', 'data', allow_duplicate=True)],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks')],
    [State('animation-state', 'data')],
    prevent_initial_call=True
)
def toggle_animation(play_clicks, pause_clicks, animation_state):
    # Find which button was clicked last
    ctx = dash.callback_context
    if not ctx.triggered:
        # No button has been clicked yet
        return True, {"playing": False}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'play-button':
        return False, {"playing": True}  # Enable animation
    else:
        return True, {"playing": False}  # Disable animation

# Callback to update time slider value during animation
@app.callback(
    Output('time-slider', 'value', allow_duplicate=True),
    [Input('animation-interval', 'n_intervals')],
    [State('time-slider', 'value'),
     State('time-slider', 'max'),
     State('time-slider', 'min'),
     State('animation-state', 'data')],
    prevent_initial_call=True
)
def update_time_slider_animation(n_intervals, current_value, max_value, min_value, animation_state):
    # If animation state is not playing, don't update
    if not animation_state or not animation_state.get("playing", False):
        raise dash.exceptions.PreventUpdate
    
    # Use the time_increment constant to ensure consistent jumps
    new_value = current_value + time_increment
    
    # Loop back to the beginning if we reach the end
    if new_value > max_value:
        new_value = min_value
    
    return new_value

# Callback to update the plot based on time slider and selected flight
@app.callback(
    [Output('drone-position-plot', 'figure'),
     Output('time-display', 'children')],
    [Input('time-slider', 'value'),
     Input('display-options', 'value'),
     Input('flight-selector', 'value')]
)
def update_plot(selected_time, display_options, csv_file):
    # Load the data for the selected flight
    df = load_flight_data(csv_file)
    
    # Calculate axis ranges for this flight
    axis_ranges = calculate_axis_ranges(df)
    
    if df.empty or 'relative_time' not in df.columns:
        # Create empty plot if no data
        fig = px.scatter_3d(
            pd.DataFrame({'x': [0], 'y': [0], 'z': [0]}),
            x='x', y='y', z='z',
            title="No drone data available"
        )
        
        # Set fixed axis ranges
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=axis_ranges['x']),
                yaxis=dict(range=axis_ranges['y']),
                zaxis=dict(range=axis_ranges['z'])
            )
        )
        return fig, "No data available"
    
    # Get a display name for the title
    if csv_file == "none":
        display_name = "Unknown Data"
    else:
        # Try to get a meaningful name
        parent_dir = os.path.basename(os.path.dirname(csv_file))
        if "Flight" in parent_dir:
            display_name = parent_dir
        else:
            display_name = os.path.basename(csv_file)
    
    # Filter data based on selected time (get closest points)
    # For each drone, find the position at the time closest to selected_time
    closest_points = []
    for drone_id in df['drone_id'].unique():
        drone_data = df[df['drone_id'] == drone_id]
        # Find the row with time closest to selected_time
        closest_idx = (drone_data['relative_time'] - selected_time).abs().idxmin()
        closest_points.append(drone_data.loc[closest_idx])
    
    # Create dataframe with the closest points
    plot_df = pd.DataFrame(closest_points)
    
    # Start with an empty figure
    fig = go.Figure()
    
    # Add scatter points for each drone
    colors = {'drone1': '#3498db', 'drone2': '#e74c3c', 'drone3': '#2ecc71'}
    default_color = '#f39c12'
    
    for i, row in plot_df.iterrows():
        drone_id = row['drone_id']
        color = colors.get(drone_id, default_color)
        
        # Add text label if enabled
        text = drone_id if 'labels' in display_options else None
        
        fig.add_trace(go.Scatter3d(
            x=[row['x']],
            y=[row['y']],
            z=[row['z']],
            mode='markers+text' if 'labels' in display_options else 'markers',
            marker=dict(
                size=12,
                color=color,
                opacity=1.0,
                symbol='circle'
            ),
            text=text,
            textposition="top center",
            name=drone_id,
            showlegend=True
        ))
    
    # Add trajectory lines if enabled
    if 'trails' in display_options:
        for drone_id in df['drone_id'].unique():
            drone_data = df[df['drone_id'] == drone_id]
            color = colors.get(drone_id, default_color)
            
            # Filter for data up to the current time
            past_data = drone_data[drone_data['relative_time'] <= selected_time]
            
            # Keep only the last 3 seconds
            time_window = 3.0  # seconds
            start_time = max(selected_time - time_window, past_data['relative_time'].min())
            recent_data = past_data[past_data['relative_time'] >= start_time]
            
            if len(recent_data) > 1:
                fig.add_trace(go.Scatter3d(
                    x=recent_data['x'],
                    y=recent_data['y'],
                    z=recent_data['z'],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=4
                    ),
                    name=f"{drone_id} trail",
                    showlegend=False
                ))
    
    # Set layout with fixed axis ranges and other customization
    fig.update_layout(
        title=f"{display_name} - Drone Positions at Time: {selected_time:.2f}s",
        scene=dict(
            xaxis=dict(
                title="X Position (m)",
                range=axis_ranges['x'],
                showgrid='grid' in display_options,
                showline=True,
                zeroline=True,
                zerolinecolor='#ccc'
            ),
            yaxis=dict(
                title="Y Position (m)",
                range=axis_ranges['y'],
                showgrid='grid' in display_options,
                showline=True,
                zeroline=True,
                zerolinecolor='#ccc'
            ),
            zaxis=dict(
                title="Z Position (m)",
                range=axis_ranges['z'],
                showgrid='grid' in display_options,
                showline=True,
                zeroline=True,
                zerolinecolor='#ccc'
            ),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Add ground plane for reference at z=0
    if 'grid' in display_options:
        x_grid = np.linspace(axis_ranges['x'][0], axis_ranges['x'][1], 2)
        y_grid = np.linspace(axis_ranges['y'][0], axis_ranges['y'][1], 2)
        x_plane, y_plane = np.meshgrid(x_grid, y_grid)
        z_plane = np.zeros(x_plane.shape)
        
        fig.add_trace(go.Surface(
            x=x_plane, y=y_plane, z=z_plane,
            colorscale=[[0, 'rgba(200, 200, 200, 0.2)'], [1, 'rgba(200, 200, 200, 0.2)']],
            showscale=False,
            name='Ground',
            showlegend=False
        ))
    
    # Format time display
    time_display = f"{selected_time:.2f} seconds"
    
    return fig, time_display

# Callback to update the animation interval based on speed selection
@app.callback(
    Output('animation-interval', 'interval'),
    [Input('speed-dropdown', 'value')]
)
def update_animation_speed(speed_factor):
    # Base interval is 500ms
    base_interval = 500
    # Adjust based on speed factor (smaller interval = faster)
    return int(base_interval / speed_factor)

# Run the app
if __name__ == '__main__':
    print("Starting Drone Position Visualizer...")
    print(f"Looking for CSV files in: /home/emeka/drone_csv_recordings")
    print(f"Found {len(available_flights)} flight datasets")
    print(f"Loaded initial data with {len(df)} rows")
    print(f"Animation will advance {time_increment} seconds at each step")
    print(f"Using fixed axis ranges: X={axis_ranges['x']}, Y={axis_ranges['y']}, Z={axis_ranges['z']}")
    app.run(debug=True, port=8050)