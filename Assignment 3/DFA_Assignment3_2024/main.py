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
    df = pd.read_csv(f'C:/Users/calvi/Documents/Projects/Data-Fusion-Architecture/Assignment 3/DFA_Assignment3_2024/{filename}')
    output_df = df[:limit] if limit > 0 else df[:]
    output_df = add_timestamps(output_df, 'timestamp_sec', 'timestamp_nanosec')
    return output_df
# Preprocessing Functions
def add_timestamps(df, sec_col, nsec_col):
    """Add a combined timestamp column and delta_time to the dataframe."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df[sec_col], unit='s') + pd.to_timedelta(df[nsec_col], unit='ns')
    df['delta_time'] = df['timestamp'].diff().dt.total_seconds()
    return df

def calculate_wheel_velocities(df):
    """Calculate linear and angular velocities from wheel data."""
    df['linear_velocity'] = (df['velocity_left'] + df['velocity_right']) * WHEEL_RADIUS / 2
    df['angular_velocity'] = (df['velocity_right'] - df['velocity_left']) * WHEEL_RADIUS / WHEEL_BASE
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
    df['delta_ticks_right'] = df['ticks_right'].diff()
    df['angular_velocity_left'] = (2 * np.pi * df['delta_ticks_left']) / (TICKS_PER_REVOLUTION * df['delta_time'])
    df['angular_velocity_right'] = (2 * np.pi * df['delta_ticks_right']) / (TICKS_PER_REVOLUTION * df['delta_time'])
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

# Load DataFrames
tf_df = load_data('tf_data.csv')
imu_df = load_data('imu_data.csv')
mouse_df = load_data('mouse_data.csv')
wheel_ticks_df = load_data('wheel_ticks_data.csv')
wheel_vels_df = load_data('wheel_vels_data.csv')

# Align DataFrames
imu_df, mouse_df, wheel_ticks_df, wheel_vels_df = align_timestamps(imu_df, mouse_df, wheel_ticks_df, wheel_vels_df)
print(len(imu_df), len(mouse_df), len(wheel_ticks_df), len(wheel_vels_df))
# Process Data
imu_df = calculate_imu_position(imu_df)
mouse_df = calculate_mouse_position(mouse_df, imu_df)
wheel_ticks_df['yaw'] = imu_df['yaw']
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
axs[1, 0].plot(imu_df['timestamp'], wheel_ticks_df['linear_velocity'], label='IMU X')
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

#%%
# UKF Parameters
n = 4  # State dimension: [x, y, linear_speed, yaw]
kappa = 0  # Scaling parameter
alpha = 1e-3  # Spread of sigma points
beta = 2  # Optimal for Gaussian distributions

# Initial State
x = np.zeros((n, 1))  # Initial state: [x, y, linear_speed, yaw]
P = np.eye(n)  # Initial covariance matrix
Q = np.eye(n) * 0.1  # Process noise covariance

# Measurement Noise Covariance
R = np.diag([
    0.5,        # Variance for position_x
    0.5,        # Variance for position_y
    0.5,      # Variance for linear_speed
    0.1       # Variance for yaw
])

# UKF Functions
def compute_sigma_points(x, P, kappa):
    """Compute sigma points."""
    n = x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))

    # Calculate square root of (n + lambda) * P
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * P)

    # Sigma points
    sigma_points[0] = x.flatten()  # Mean state
    for i in range(n):
        sigma_points[i + 1] = x.flatten() + sqrt_matrix[:, i]
        sigma_points[n + i + 1] = x.flatten() - sqrt_matrix[:, i]

    return sigma_points, lambda_

def predict_sigma_points(sigma_points, dt):
    """Predict sigma points through the process model."""
    n = sigma_points.shape[1]
    sigma_points_pred = np.zeros_like(sigma_points)

    for i in range(sigma_points.shape[0]):
        # Extract state variables
        px, py, v, yaw = sigma_points[i]

        # Predict next state
        px += v * np.cos(yaw) * dt
        py += v * np.sin(yaw) * dt
        sigma_points_pred[i] = [px, py, v, yaw]  # v and yaw remain unchanged

    return sigma_points_pred

def predict_mean_and_covariance(sigma_points, Wm, Wc, Q):
    """Predict the mean and covariance from sigma points."""
    x_pred = np.sum(Wm[:, None] * sigma_points, axis=0)
    P_pred = Q.copy()
    for i in range(sigma_points.shape[0]):
        diff = sigma_points[i] - x_pred
        # diff[3] = np.arctan2(np.sin(diff[3]), np.cos(diff[3]))  # Normalize yaw
        P_pred += Wc[i] * np.outer(diff, diff)
    return x_pred, P_pred

def unscented_transform(sigma_points, Wm, Wc, R, z_func):
    """Transform sigma points to measurement space."""
    sigma_points_meas = np.array([z_func(sigma) for sigma in sigma_points])
    z_pred = np.sum(Wm[:, None] * sigma_points_meas, axis=0)

    # Covariance in measurement space
    S = R.copy()
    for i in range(sigma_points_meas.shape[0]):
        diff = sigma_points_meas[i] - z_pred
        # diff[3] = np.arctan2(np.sin(diff[3]), np.cos(diff[3]))  # Normalize yaw
        S += Wc[i] * np.outer(diff, diff)

    return z_pred, S, sigma_points_meas

def measurement_function(sigma):
    """Map sigma points from state space to measurement space."""
    px, py, v, yaw = sigma
    return np.array([px, py, v, yaw])

# UKF Loop
for i in range(1, len(imu_df)):
    # Time step
    dt = imu_df['delta_time'].iloc[i]
    if np.isnan(dt) or dt <= 0:
        continue

    # Compute sigma points
    sigma_points, lambda_ = compute_sigma_points(x.flatten(), P, kappa)

    # Weights
    n_sigma = 2 * n + 1
    Wm = np.full(n_sigma, 0.5 / (n + lambda_))
    Wm[0] = lambda_ / (n + lambda_)
    Wc = Wm.copy()
    Wc[0] += 1 - alpha**2 + beta

    # Prediction step
    sigma_points_pred = predict_sigma_points(sigma_points, dt)
    x_pred, P_pred = predict_mean_and_covariance(sigma_points_pred, Wm, Wc, Q)

    # Measurement vector
    z = np.array([
        wheel_ticks_df['position_x'].iloc[i],
        wheel_ticks_df['position_y'].iloc[i],
        wheel_vels_df['linear_velocity'].iloc[i],
        imu_df['yaw'].iloc[i]
    ])

    # Measurement update
    z_pred, S, sigma_points_meas = unscented_transform(sigma_points_pred, Wm, Wc, R, measurement_function)
    cross_covariance = np.zeros((n, z.shape[0]))
    for j in range(n_sigma):
        diff_x = sigma_points_pred[j] - x_pred
        diff_x[3] = np.arctan2(np.sin(diff_x[3]), np.cos(diff_x[3]))  # Normalize yaw
        diff_z = sigma_points_meas[j] - z_pred
        diff_z[3] = np.arctan2(np.sin(diff_z[3]), np.cos(diff_z[3]))  # Normalize yaw
        cross_covariance += Wc[j] * np.outer(diff_x, diff_z)

    # Kalman gain
    K = cross_covariance @ np.linalg.inv(S)

    # Update step
    diff_z = z - z_pred
    diff_z[3] = np.arctan2(np.sin(diff_z[3]), np.cos(diff_z[3]))  # Normalize yaw
    x = x_pred + K @ diff_z
    P = P_pred - K @ S @ K.T

    # Save results
    imu_df.loc[i, 'ukf_x'] = x[0]
    imu_df.loc[i, 'ukf_y'] = x[1]

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(wheel_vels_df['position_x'], wheel_vels_df['position_y'], label='Raw wheel_vels_df Position', color='red')
plt.plot(tf_df['translation_x'], tf_df['translation_y'], label='TF Ground Truth', color='green')
plt.plot(wheel_ticks_df['position_x'], wheel_ticks_df['position_y'], label='Raw Wheel Tick Truth', color='orange')
# plt.plot(imu_df['ukf_x'], imu_df['ukf_y'], label='UKF Position', color='blue')
plt.xlim(1.8, 2.2)
plt.ylim(-0.1, 0.1)
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Unscented Kalman Filter (UKF) Position')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Constants
DELTA_T = 0.1  # Time step (seconds)
AREA_LIMIT = 3.0  # 3x3 square
NUM_PARTICLES = 500  # Number of particles
MOTION_NOISE = [0.1, 0.1, np.deg2rad(1)]  # Noise in [x, y, theta]
LANDMARKS = [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]]  # Landmarks in the area
def load_tf_data(filename):
    """Load tf_data (ground truth) from a CSV file."""
    # df = pd.read_csv(filename)
    imu_df['ukf_x'][0] = 0.0
    imu_df['ukf_y'][0] = 0.0
    imu_df['yaw'][0] = 0.0
    return imu_df['ukf_x'], imu_df['ukf_y'], imu_df['yaw']
    # df = pd.read_csv(filename)
    # return df['translation_x'], df['translation_y'], df['rotation_z']

# Particle Filter Class
class ParticleFilter:
    def __init__(self, num_particles, area_limit):
        self.num_particles = num_particles
        self.area_limit = area_limit
        # Initialize particles uniformly within the area
        self.particles = np.zeros((num_particles, 3))  # [x, y, theta]
        self.particles[:, 0] = np.random.uniform(0, area_limit, num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, area_limit, num_particles)  # y
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)  # theta
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input):
        """Propagate particles using the motion model with noise."""
        v, omega = control_input  # Linear and angular velocity
        noise = np.random.normal(0, MOTION_NOISE, (self.num_particles, 3))
        self.particles[:, 2] += omega * DELTA_T + noise[:, 2]  # Update theta
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
        self.particles[:, 0] += (v * np.cos(self.particles[:, 2]) * DELTA_T + noise[:, 0])  # Update x
        self.particles[:, 1] += (v * np.sin(self.particles[:, 2]) * DELTA_T + noise[:, 1])  # Update y
        # Clamp particles to the area limit
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.area_limit)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.area_limit)

    def update(self, measurements, measurement_noise, landmarks):
        """Update particle weights based on measurements to landmarks."""
        for i, landmark in enumerate(landmarks):
            # Calculate expected distances to this landmark for all particles
            expected_distances = np.sqrt((self.particles[:, 0] - landmark[0])**2 + 
                                         (self.particles[:, 1] - landmark[1])**2)
            # Actual distance measurement for this landmark
            actual_distance = measurements[i]
            
            # Calculate weights based on Gaussian likelihood
            self.weights *= np.exp(-0.5 * ((expected_distances - actual_distance) / measurement_noise[0])**2)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on their weights."""
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights

    def estimate(self):
        """Estimate the state as the weighted mean of the particles."""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        covariance = np.cov(self.particles.T, aweights=self.weights)
        return mean, covariance

# Simulate Landmark Measurements
def simulate_landmark_measurements(ground_truth, landmarks, noise_std):
    """Simulate measurements from the ground truth position to landmarks."""
    measurements = []
    for landmark in landmarks:
        distance = np.sqrt((ground_truth[0] - landmark[0])**2 + (ground_truth[1] - landmark[1])**2)
        noisy_distance = distance + np.random.normal(0, noise_std)
        measurements.append(noisy_distance)
    return measurements

# Load tf_data (assuming the file is named 'tf_data.csv')
tf_x, tf_y, tf_theta = load_tf_data('tf_data.csv')
ground_truth = np.vstack((tf_x, tf_y, tf_theta)).T

# Initialize Particle Filter
pf = ParticleFilter(NUM_PARTICLES, AREA_LIMIT)

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 8))
landmark_scatter = ax.scatter([l[0] for l in LANDMARKS], [l[1] for l in LANDMARKS],
                               color='green', label='Landmarks', s=100)
particles_scatter = ax.scatter([], [], s=[], alpha=0.4, label='Particles')
ground_truth_scatter = ax.scatter([], [], color='red', label='Ground Truth', s=100)
predicted_scatter = ax.scatter([], [], color='blue', marker='x', label='Prediction', s=100)
ax.set_xlim(-0.5, AREA_LIMIT + 0.5)
ax.set_ylim(-0.5, AREA_LIMIT + 0.5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.legend()
ax.grid()

# Animation update function
def update(frame):
    control_input = [0.5, np.deg2rad(5)]  # Example control input
    measurement = simulate_landmark_measurements(ground_truth[frame], LANDMARKS, noise_std=0.2)

    # Particle filter steps
    pf.predict(control_input)
    pf.update(measurement, measurement_noise=[0.2], landmarks=LANDMARKS)
    pf.resample()

    # Estimate the state
    estimated_mean, _ = pf.estimate()

    particles_scatter.set_offsets(pf.particles[:, :2])
    particles_scatter.set_sizes(pf.weights * 1000)
    ground_truth_scatter.set_offsets([ground_truth[frame, 0], ground_truth[frame, 1]])
    predicted_scatter.set_offsets([estimated_mean[0], estimated_mean[1]])
    return particles_scatter, ground_truth_scatter, predicted_scatter

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(ground_truth) - 1, interval=200, blit=True)

plt.show()

# Save as GIF
ani.save('particle_filter_animation_with_prediction.gif', writer='pillow', fps=10)

print("GIF saved as 'particle_filter_animation_with_prediction.gif'")

# %%
