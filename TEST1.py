import numpy as np
import time

# Function for spherical to cartesian conversion
def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

# Doppler correlation function
def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

# Range gating function
def range_gate(distance, range_threshold):
    return distance < range_threshold

# Function to initialize and update tracks
def initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold):
    tracks = []
    track_ids = {}
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()

    # We add timestamps to each measurement.
    for measurement in measurements:
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        # Flag to determine if measurement was assigned
        assigned = False

        for track_id, track in enumerate(tracks):
            last_measurement = track[-1]
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            last_doppler = last_measurement[3]
            last_time = last_measurement[4]

            # Calculate distance and check conditions
            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)
            range_satisfied = range_gate(distance, range_threshold)

            # Time difference check
            time_diff = measurement_time - last_time

            if doppler_correlated and range_satisfied and time_diff <= time_threshold:
                if track_id not in firm_ids:
                    if track_id in tentative_ids:
                        hit_counts[track_id] += 1
                        miss_counts[track_id] = 0
                        if hit_counts[track_id] >= firm_threshold:
                            firm_ids.add(track_id)
                            print(f"Track ID {track_id} is now firm.")
                    else:
                        tentative_ids[track_id] = True
                        hit_counts[track_id] = 1
                        miss_counts[track_id] = 0
                tracks[track_id].append(measurement)
                print(f"Measurement {measurement} assigned to Track ID {track_id}: Doppler and Range conditions satisfied.")
                assigned = True
                break
            elif doppler_correlated or range_satisfied and time_diff <= time_threshold:
                # Prefer Doppler correlation if both conditions are not met
                if doppler_correlated:
                    if track_id not in firm_ids:
                        if track_id in tentative_ids:
                            hit_counts[track_id] += 1
                            miss_counts[track_id] = 0
                            if hit_counts[track_id] >= firm_threshold:
                                firm_ids.add(track_id)
                                print(f"Track ID {track_id} is now firm.")
                        else:
                            tentative_ids[track_id] = True
                            hit_counts[track_id] = 1
                            miss_counts[track_id] = 0
                    tracks[track_id].append(measurement)
                    print(f"Measurement {measurement} assigned to Track ID {track_id}: Doppler condition satisfied.")
                    assigned = True
                    break
                elif range_satisfied:
                    if track_id not in firm_ids:
                        if track_id in tentative_ids:
                            hit_counts[track_id] += 1
                            miss_counts[track_id] = 0
                            if hit_counts[track_id] >= firm_threshold:
                                firm_ids.add(track_id)
                                print(f"Track ID {track_id} is now firm.")
                        else:
                            tentative_ids[track_id] = True
                            hit_counts[track_id] = 1
                            miss_counts[track_id] = 0
                    tracks[track_id].append(measurement)
                    print(f"Measurement {measurement} assigned to Track ID {track_id}: Range condition satisfied.")
                    assigned = True
                    break

        if not assigned:
            # Create a new track
            track_id = len(tracks)
            tracks.append([measurement])
            track_ids[track_id] = track_id
            miss_counts[track_id] = 0
            hit_counts[track_id] = 1
            tentative_ids[track_id] = True
            print(f"Measurement {measurement} initiated a new Track ID {track_id}.")

        # Increment miss count for all tracks that were not assigned this measurement
        for track_id in range(len(tracks)):
            if track_id not in firm_ids and not assigned:
                if track_id in miss_counts:
                    miss_counts[track_id] += 1
                    if miss_counts[track_id] > firm_threshold:
                        print(f"Track ID {track_id} has too many misses and will be removed.")
                        # Remove track if too many misses
                        tracks[track_id] = []

    return tracks, track_ids, miss_counts, hit_counts, firm_ids

# Sample measurements (azimuth, elevation, range, doppler, timestamp in seconds)
sample_measurements = [
    (10, 5, 100, 5, 0),
    (12, 6, 105, 6, 1),
    (9, 4, 98, 4.5, 2),
    (50, 20, 500, 50, 3),
    # Continue adding measurements with timestamps...
]

# Parameters for gating
doppler_threshold = 2.0  # Doppler gate threshold
range_threshold = 10.0   # Range gate threshold in Cartesian distance
firm_threshold = 3       # Number of continuous hits needed to firm a track
time_threshold = 2.0     # Time window threshold in seconds

# Initialize tracks
tracks, track_ids, miss_counts, hit_counts, firm_ids = initialize_tracks(
    sample_measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold
)

# Output the tracks and their associated measurements
for track_id, track in enumerate(tracks):
    print(f"Track ID {track_id}:")
    for measurement in track:
        print(f"  Measurement: {measurement}")
    print(f"  Hits: {hit_counts.get(track_id, 0)}, Misses: {miss_counts.get(track_id, 0)}")
    if track_id in firm_ids:
        print(f"  Track ID {track_id} is firm.")
    else:
        print(f"  Track ID {track_id} is tentative.")
