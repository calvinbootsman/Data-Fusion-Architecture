#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Constants
WHEEL_RADIUS = 0.072 / 2  # Radius of the wheel (meters)
WHEEL_BASE = 0.235        # Distance between wheels (meters)
TICKS_PER_REVOLUTION = 508.8
DATA_LIMIT = 953          # Set to -1 for no limit on data

# Load data
def load_data(filename, limit=DATA_LIMIT):
    """Loads a CSV file and applies a data limit if specified."""
    df = pd.read_csv(f'{filename}')
    output_df = df[:limit] if limit > 0 else df[:]
    output_df = add_timestamps(output_df, 'timestamp_sec', 'timestamp_nanosec')
    return output_df[10:]
# Preprocessing Functions
def add_timestamps(df, sec_col, nsec_col):
    """Add a combined timestamp column and delta_time to the dataframe."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df[sec_col], unit='s') + pd.to_timedelta(df[nsec_col], unit='ns')
    df['delta_time'] = df['timestamp'].diff().dt.total_seconds()
    df.loc[0, 'delta_time'] = 0
    return df

def calculate_wheel_velocities(df):
    """Calculate linear and angular velocities from wheel data."""
    df['linear_velocity'] = (df['velocity_left'] + df['velocity_right']) * WHEEL_RADIUS / 2
    df.loc[0, 'linear_velocity'] = 0
    df['angular_velocity'] = (df['velocity_right'] - df['velocity_left']) * WHEEL_RADIUS / WHEEL_BASE
    df.loc[0, 'angular_velocity'] = 0
    df['delta_yaw'] = df['angular_velocity'] * df['delta_time']

    if 'yaw' not in df.columns:
        df['yaw'] = df['delta_yaw'].cumsum()
    
    df['speed_x'] = df['linear_velocity'] * np.cos(df['yaw'])
    df['speed_y'] = df['linear_velocity'] * np.sin(df['yaw'])
    df['displacement_x'] = df['speed_x'] * df['delta_time']
    df['displacement_y'] = df['speed_y'] * df['delta_time']
    df['position_x'] = df['displacement_x'].cumsum()
    df['position_y'] = df['displacement_y'].cumsum()
    return df

def calculate_wheel_ticks(df):
    """Calculate velocities and positions using wheel tick data."""
    print(TICKS_PER_REVOLUTION)
    df['delta_ticks_left'] = df['ticks_left'].diff()
    df.loc[0, 'delta_ticks_left'] = 0
    df['delta_ticks_right'] = df['ticks_right'].diff()
    df.loc[0, 'delta_ticks_right'] = 0
    df['angular_velocity_left'] = (2 * np.pi * df['delta_ticks_left']) / (TICKS_PER_REVOLUTION * df['delta_time'])
    df.loc[0, 'angular_velocity_left'] = 0
    df['angular_velocity_right'] = (2 * np.pi * df['delta_ticks_right']) / (TICKS_PER_REVOLUTION * df['delta_time'])
    df.loc[0, 'angular_velocity_right'] = 0
    df['velocity_left'] = df['angular_velocity_left']
    df['velocity_right'] = df['angular_velocity_right']
    return calculate_wheel_velocities(df)

def calculate_imu_position(df):
    """Calculate positions from IMU data."""
    # Calculate yaw from quaternion (as before)
    w = df['orientation_w']
    x = df['orientation_x']
    y = df['orientation_y']
    z = df['orientation_z']
    
    df['yaw_wrapped'] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    df['yaw'] = np.unwrap(df['yaw_wrapped'])

    df['acceleration_x'] = df['linear_acceleration_x'] * np.cos(df['yaw'])
    df['acceleration_y'] = df['linear_acceleration_y'] * np.sin(df['yaw'])

    # Reconstruct the yaw
    df['velocity_x'] = (df['linear_acceleration_x'] * df['delta_time']).cumsum()
    df['velocity_y'] = (df['linear_acceleration_y'] * df['delta_time']).cumsum()

    # Calculate total velocity magnitude
    df['linear_velocity'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    
    df['speed_x'] = df['linear_velocity'] * np.cos(df['yaw']) #df['speed_imu_x']
    df['speed_y'] = df['linear_velocity'] * np.sin(df['yaw'])#df['speed_imu_y']
    df['displacement_x'] = df['speed_x'] * df['delta_time']
    df['displacement_y'] = df['speed_y'] * df['delta_time']
    df['position_x'] = df['displacement_x'].cumsum()
    df['position_y'] = df['displacement_y'].cumsum()

    return df

def calculate_mouse_position(mouse_df, imu_df):
    """Calculate mouse-based positions, rotated using IMU yaw."""

    mouse_df['delta_x'] = mouse_df['integrated_x'].diff() *-1
    mouse_df['delta_y'] = mouse_df['integrated_y'].diff() 
    mouse_df['rotated_delta_x'] = mouse_df['delta_x'] * np.cos(imu_df['yaw'])
    mouse_df['rotated_delta_y'] = mouse_df['delta_y'] * np.sin(imu_df['yaw'])
    mouse_df['position_x'] = mouse_df['rotated_delta_x'].cumsum()
    mouse_df['position_y'] = mouse_df['rotated_delta_y'].cumsum()
    return mouse_df

def align_timestamps(df1, df2, df3, df4):
    """Align five dataframes based on timestamps using pd.merge_asof."""
    # Assuming df1 has the most complete/accurate timestamps, use it as the base
    base_timestamps = df1['timestamp']

    # Align each DataFrame to the base timestamps
    df2 = pd.merge_asof(df2, base_timestamps, on='timestamp', direction='nearest')
    df3 = pd.merge_asof(df3, base_timestamps, on='timestamp', direction='nearest')
    df4 = pd.merge_asof(df4, base_timestamps, on='timestamp', direction='nearest')

    return df1, df2, df3, df4

def quaternion_cords(df):
    w = df['rotation_w']
    x = df['rotation_x']
    y = df['rotation_y']
    z = df['rotation_z']
    
    df['yaw_wrapped'] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    df['yaw'] = np.unwrap(df['yaw_wrapped'])

    return df

# Load DataFrames
tf_df = load_data('tf_data.csv')
imu_df = load_data('imu_data.csv')
mouse_df = load_data('mouse_data.csv')
wheel_ticks_df = load_data('wheel_ticks_data.csv')
wheel_vels_df = load_data('wheel_vels_data.csv')

tf_df = quaternion_cords(tf_df)

# Align DataFrames
imu_df, mouse_df, wheel_ticks_df, wheel_vels_df = align_timestamps(imu_df, mouse_df, wheel_ticks_df, wheel_vels_df)
print(len(imu_df), len(mouse_df), len(wheel_ticks_df), len(wheel_vels_df))
# Process Data
imu_df = calculate_imu_position(imu_df)
mouse_df = calculate_mouse_position(mouse_df, imu_df)
wheel_ticks_df['yaw'] = imu_df['yaw'] + 0.00029
wheel_ticks_df = calculate_wheel_ticks(wheel_ticks_df)
wheel_vels_df['yaw'] = imu_df['yaw']
wheel_vels_df = calculate_wheel_velocities(wheel_vels_df)

# Plot Results
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Position X vs. Position Y
axs[0, 0].plot(tf_df['translation_x'], tf_df['translation_y'], label='TF Translation', linewidth=2)
axs[0, 0].plot(wheel_vels_df['position_x'], wheel_vels_df['position_y'], label='Wheel Velocities', linestyle='--')
axs[0, 0].plot(imu_df['position_x'], imu_df['position_y'], label='IMU Position', linestyle=':')
axs[0, 0].plot(wheel_ticks_df['position_x'], wheel_ticks_df['position_y'], label='Wheel Ticks', linestyle='-.')
# axs[0, 0].plot(mouse_df['position_x'], mouse_df['position_y'], label='Mouse Translation', linewidth=1)
axs[0, 0].set_xlabel('Position X (meters)')
axs[0, 0].set_ylabel('Position Y (meters)')
axs[0, 0].set_title('Position X vs. Position Y')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot 2: Difference in Timestamps between IMU and Wheel Ticks Data
timestamp_diff = imu_df['timestamp'].values - wheel_vels_df['timestamp'].values
axs[0, 1].plot(wheel_vels_df['yaw'], label='Wheel Velocities Yaw')
axs[0, 1].plot(imu_df['yaw'], label='IMU Yaw')
axs[0, 1].plot(wheel_ticks_df['yaw'], label='Wheel Ticks Yaw')
axs[0, 1].set_xlabel('Index')
axs[0, 1].set_ylabel('Yaw (radians)')
axs[0, 1].set_title('Yaw Comparison')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Subplot 3: IMU Linear Acceleration X over Time
data_df = imu_df[:170]
print("Variance of angular_velocity_z:", data_df['angular_velocity_z'].var())
print("Variance of linear_acceleration_x:", data_df['linear_acceleration_x'].var())
print("Variance of linear_acceleration_y:", data_df['linear_acceleration_y'].var())
print("Variance of orientation_z:", data_df['orientation_z'].var())
data_df = imu_df[200:270]
print("Variance of angular_velocity_z:", data_df['angular_velocity_z'].var())
print("Variance of linear_acceleration_x:", data_df['linear_acceleration_x'].var())
print("Variance of linear_acceleration_y:", data_df['linear_acceleration_y'].var())
print("Variance of orientation_z:", data_df['orientation_z'].var())
axs[1, 0].plot( data_df['linear_acceleration_x'], label='IMU X')
axs[1, 0].plot( wheel_ticks_df['linear_velocity'][:800], label='IMU X')
# axs[1, 0].plot(imu_df['timestamp'], imu_df['velocity_y'], label='IMU Y')
# axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_ylabel('Linear Acceleration (m/s^2)')
axs[1, 0].set_title('IMU Linear Acceleration over Time')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Subplot 4: IMU Yaw over Time
axs[1, 1].plot(imu_df['timestamp'], imu_df['yaw'], label='IMU Yaw', color='purple')
axs[1, 1].plot(imu_df['timestamp'], np.deg2rad(imu_df['orientation_z'] * 360), label='IMU Yaw', color='orange')
axs[1, 1].set_xlabel('Timestamp')
axs[1, 1].set_ylabel('Yaw (radians)')
axs[1, 1].set_title('IMU Yaw over Time')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# %%
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

def f_x(state, dt):
    """
    State transition function for the robot.

    States:
        x[0]: x position
        x[1]: y position
        x[2]: yaw
        x[3]: linear velocity
        x[4]: x acceleration
        x[5]: y acceleration
        x[6]: yaw rate 
    """
    x, y, yaw, v, a_x, a_y, yaw_rate = state

    new_x = x + v * np.cos(yaw) * dt + 0.5 * a_x * dt**2
    new_y = y + v * np.sin(yaw) * dt + 0.5 * a_y * dt**2
    new_yaw = yaw + yaw_rate * dt
    # Calculate the unit vector in the direction of the current velocity (using yaw)
    v_unit_x = np.cos(yaw)
    v_unit_y = np.sin(yaw)

    # Project the acceleration onto the velocity unit vector
    a_parallel = a_x * v_unit_x + a_y * v_unit_y

    # Update the velocity
    new_v = v + a_parallel * dt
    # new_v = v + np.sqrt(a_x**2 + a_y**2) * dt
    return np.array([new_x, new_y, new_yaw, new_v, a_x, a_y, yaw_rate])

# 2. Measurement Function
def h_x(state):
    """
    Measurement function. We measure the linear velocity, yaw, acceleration_x, acceleration_y and yaw rate

    States:
        x[0]: x position
        x[1]: y position
        x[2]: yaw
        x[3]: linear velocity
        x[4]: x acceleration
        x[5]: y acceleration
        x[6]: yaw rate
    """
    return np.array([state[3], state[2], state[4], state[5], state[6]])  # [linear velocity, yaw, acceleration_x, acceleration_y, yaw_rate]

# 3. Sigma Points and Weights
points = MerweScaledSigmaPoints(n=7, alpha=0.1, beta=2.0, kappa=-1)

# 4. Create UKF
ukf = UKF(dim_x=7, dim_z=5, fx=f_x, hx=h_x, dt=imu_df['delta_time'].iloc[0], points=points)

# 5. Initialize State
# (x, y, yaw, v, a_x, a_y, yaw_rate)
ukf.x = np.array([0., 0., imu_df['yaw'].iloc[0], wheel_vels_df['linear_velocity'].iloc[0], imu_df['acceleration_x'].iloc[0], imu_df['acceleration_y'].iloc[0], imu_df['angular_velocity_z'].iloc[0]])  
# x, y, yaw, v, a_x, a_y, yaw_rate
ukf.P = np.diag([
    0.001 ** 2,  # x position variance (increase if you have no prior knowledge of initial position)
    0.001 ** 2,  # y position variance (increase if you have no prior knowledge of initial position)
    np.deg2rad(5)**2,  # Yaw variance (e.g., standard deviation of 5 degrees)
    0.01**2,  # Linear velocity variance
    data_df['linear_acceleration_x'][:170].var(),  # a_x variance (from stationary data)
    data_df['linear_acceleration_y'][:170].var(),  # a_y variance (from stationary data)
    0.001**2   # Yaw rate variance
])

# 6. Measurement Noise
ukf.R = np.diag([
    0.004 ** 2,  #  linear velocity variance from encoders (placeholder - needs to be determined)
    imu_df['orientation_z'][:170].var(),  # Increased yaw variance (stationary)
    imu_df['linear_acceleration_x'][:170].var(),    # a_x variance (stationary)
    imu_df['linear_acceleration_y'][:170].var(),    # a_y variance (stationary)
    imu_df['angular_velocity_z'][:170].var()  # Increased yaw rate variance (stationary)
])

print(imu_df['linear_acceleration_x'][:170].var())

# 7. Process Noise
ukf.Q = np.diag([
    0.05 ** 2,  # x position process noise variance
    0.05 ** 2,  # y position process noise variance
    np.deg2rad(1) ** 2,  # Yaw process noise variance 
    0.5 ** 2,  # linear velocity process noise variance
    (1 * 1e-2) ** 2,   # a_x process noise variance
    (1* 1e-2) ** 2 ,   # a_y process noise variance
    1e-2 ** 2   # Yaw rate process noise variance
])

# 8. Run the Filter
z_combined = np.zeros((len(imu_df), 5))
z_combined[:, 0] = wheel_ticks_df['linear_velocity'].values  # Use linear velocity from wheel velocities
z_combined[:, 1] = imu_df['yaw'].values
z_combined[:, 2] = imu_df['acceleration_x'].values
z_combined[:, 3] = imu_df['acceleration_y'].values
z_combined[:, 4] = imu_df['angular_velocity_z'].values

# Prepare 'dts' to be used in batch_filter
dts = imu_df['delta_time'].values

# Run the filter with the combined measurements
mu, cov = ukf.batch_filter(z_combined, dts=dts)

# 9. Extract Results
ukf_x, ukf_y, ukf_yaw, ukf_v, ukf_a_x, ukf_a_y, ukf_yaw_rate = mu[:, 0], mu[:, 1], mu[:, 2], mu[:, 3], mu[:, 4], mu[:, 5], mu[:, 6]

# 10. Plot Results
fig, axs = plt.subplots(2, 2, figsize=(15, 8))

# Subplot 1: Robot Path Estimation
axs[0, 0].plot(wheel_ticks_df['position_x'], wheel_ticks_df['position_y'], label='Wheel Ticks Odometry', linestyle='--')
axs[0, 0].plot(wheel_vels_df['position_x'], wheel_vels_df['position_y'], label='Wheel Vels Odometry', linestyle='-.')
axs[0, 0].plot(ukf_x, ukf_y, label='UKF Estimated Path', linestyle='-')
axs[0, 0].plot(tf_df['translation_x'], tf_df['translation_y'], label='TF data', linestyle='-.')
axs[0, 0].set_xlabel('X Position (m)')
axs[0, 0].set_ylabel('Y Position (m)')
axs[0, 0].set_title('Robot Path Estimation')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot 2: UKF Linear Velocity
axs[0,1].plot(ukf_v, label='UKF Linear Velocity', linestyle='-')
axs[0,1].plot(wheel_ticks_df['linear_velocity'], label='Wheel ticks Linear Velocity', linestyle='-')
axs[0,1].set_xlabel('Timestamp')
axs[0,1].set_ylabel('Linear Velocity (m/s)')
axs[0,1].set_title('UKF Linear Velocity over Time')
axs[0,1].legend()
axs[0,1].grid(True)

# Subplot 3: UKF Acceleration X
axs[1, 0].plot(ukf_a_x, label='UKF Acceleration X', linestyle='-')
axs[1, 0].plot(imu_df['acceleration_x'], label='IMU Acceleration X', linestyle='-')
# axs[1, 0].set_ylim(-0.1, 0.100)
axs[1, 0].set_xlim(250, 400)
axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_ylabel('Acceleration X (m/s^2)')
axs[1, 0].set_title('UKF Acceleration X over Time')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Subplot 3: UKF Acceleration Y
axs[1, 1].plot(ukf_a_y, label='UKF Acceleration Y', linestyle='-')
axs[1, 1].plot(imu_df['acceleration_y'], label='IMU Acceleration Y', linestyle='-')
axs[1, 1].set_xlim(250, 400)
axs[1, 1].set_xlabel('Timestamp')
axs[1, 1].set_ylabel('Acceleration Y (m/s^2)')
axs[1, 1].set_title('UKF Acceleration Y over Time')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

plt.plot(wheel_ticks_df['position_x'], wheel_ticks_df['position_y'], label='Wheel Ticks Odometry', linestyle='--')
# plt.plot(wheel_vels_df['position_x'], wheel_vels_df['position_y'], label='Wheel Vels Odometry', linestyle='-.')
plt.plot(ukf_x, ukf_y, label='UKF Estimated Path', linestyle='-')
plt.plot(tf_df['translation_x'], tf_df['translation_y'], label='TF data', linestyle='-.')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Path Estimation')
plt.legend()
plt.grid(True)

plt.show()

# %%
