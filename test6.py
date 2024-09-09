import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QFileDialog, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

# Your functions here (sph2cart, doppler_correlation, range_gate, get_next_track_id, release_track_id, 
# initialize_tracks, load_measurements_from_csv, select_initiation_mode)

def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def range_gate(distance, range_threshold):
    return distance < range_threshold

def get_next_track_id(track_id_list):
    for idx, track in enumerate(track_id_list):
        if track['state'] == 'free':
            track_id_list[idx]['state'] = 'occupied'
            return track['id'], idx
    new_id = len(track_id_list) + 1
    track_id_list.append({'id': new_id, 'state': 'occupied'})
    return new_id, len(track_id_list) - 1

def release_track_id(track_id_list, idx):
    track_id_list[idx]['state'] = 'free'

def initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()

    for i, measurement in enumerate(measurements):
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        assigned = False

        for track_id, track in enumerate(tracks):
            if not track:
                continue
            
            last_measurement = track[-1]
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            last_doppler = last_measurement[3]
            last_time = last_measurement[4]

            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)
            range_satisfied = range_gate(distance, range_threshold)
            time_diff = measurement_time - last_time

            if doppler_correlated and range_satisfied and time_diff <= time_threshold:
                if track_id not in firm_ids:
                    if track_id in tentative_ids:
                        hit_counts[track_id] += 1
                        miss_counts[track_id] = 0
                        if hit_counts[track_id] >= firm_threshold:
                            firm_ids.add(track_id)
                    else:
                        tentative_ids[track_id] = True
                        hit_counts[track_id] = 1
                        miss_counts[track_id] = 0
                tracks[track_id].append(measurement)
                assigned = True
                break

        if not assigned:
            new_track_id, new_track_idx = get_next_track_id(track_id_list)
            tracks.append([measurement])
            miss_counts[new_track_idx] = 0
            hit_counts[new_track_idx] = 1
            tentative_ids[new_track_idx] = True

        for track_id in range(len(tracks)):
            if track_id not in firm_ids and not assigned:
                if track_id in miss_counts:
                    miss_counts[track_id] += 1
                    if miss_counts[track_id] > firm_threshold:
                        tracks[track_id] = []
                        release_track_id(track_id_list, track_id)

    return tracks, track_id_list, miss_counts, hit_counts, firm_ids

def load_measurements_from_csv(file_path):
    df = pd.read_csv(file_path)
    measurements = []

    for i in range(len(df)):
        doppler = 0 if i == 0 else (df['range'][i] - df['range'][i - 1]) / (df['timestamp'][i] - df['timestamp'][i - 1])
        measurements.append((df['azimuth'][i], df['elevation'][i], df['range'][i], doppler, df['timestamp'][i]))

    return measurements

def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid initiation mode. Choose '3-state', '5-state', or '7-state'.")

class TrackApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle('Track Initialization')
        self.setGeometry(100, 100, 600, 500)
        
        # Set color palette
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                border-radius: 5px;
                padding: 10px;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QComboBox {
                background-color: #34495e;
                color: white;
            }
            QLineEdit {
                background-color: #34495e;
                color: white;
            }
            QTextEdit {
                background-color: #34495e;
                color: white;
            }
        """)
        
        # Layout
        layout = QVBoxLayout()
        
        # File selection
        self.file_label = QLabel('Select CSV File:')
        layout.addWidget(self.file_label)
        self.file_button = QPushButton('Browse')
        self.file_button.clicked.connect(self.browse_file)
        layout.addWidget(self.file_button)
        
        # Initiation mode selection
        self.mode_label = QLabel('Select Initiation Mode:')
        layout.addWidget(self.mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['3-state', '5-state', '7-state'])
        layout.addWidget(self.mode_combo)
        
        # Doppler Threshold
        self.doppler_label = QLabel('Enter Doppler Threshold:')
        layout.addWidget(self.doppler_label)
        self.doppler_input = QLineEdit()
        layout.addWidget(self.doppler_input)
        
        # Range Threshold
        self.range_label = QLabel('Enter Range Threshold:')
        layout.addWidget(self.range_label)
        self.range_input = QLineEdit()
        layout.addWidget(self.range_input)
        
        # Time Threshold
        self.time_label = QLabel('Enter Time Threshold:')
        layout.addWidget(self.time_label)
        self.time_input = QLineEdit()
        layout.addWidget(self.time_input)
        
        # Execute button
        self.execute_button = QPushButton('Initialize Tracks')
        self.execute_button.clicked.connect(self.execute_track_initialization)
        layout.addWidget(self.execute_button)
        
        # Output text box
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        # Set layout
        self.setLayout(layout)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            self.file_path = file_path
    
    def execute_track_initialization(self):
        try:
            # Get inputs
            file_path = getattr(self, 'file_path', None)
            if not file_path:
                self.output_text.append('Please select a CSV file.')
                return
            
            doppler_threshold = float(self.doppler_input.text())
            range_threshold = float(self.range_input.text())
            time_threshold = float(self.time_input.text())
            mode = self.mode_combo.currentText()
            
            # Select initiation mode
            firm_threshold = select_initiation_mode(mode)
            
            # Load measurements from the selected CSV file
            measurements = load_measurements_from_csv(file_path)
            
            # Initialize tracks
            tracks, track_id_list, miss_counts, hit_counts, firm_ids = initialize_tracks(
                measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold
            )
            
            # Display output in the text box
            output = ''
            for track_id, track in enumerate(tracks):
                if track:
                    output += f"Track ID {track_id + 1}:\n"
                    for measurement in track:
                        output += f"  Measurement: {measurement}\n"
                    output += f"  Hits: {hit_counts.get(track_id, 0)}, Misses: {miss_counts.get(track_id, 0)}\n"
                    if track_id in firm_ids:
                        output += f"  Track ID {track_id + 1} is firm.\n"
                    else:
                        output += f"  Track ID {track_id + 1} is tentative.\n"
            
            for idx, track_info in enumerate(track_id_list):
                output += f"Track ID {track_info['id']} is {track_info['state']}.\n"
            
            self.output_text.append(output)  # Append new results to the output box
        except Exception as e:
            self.output_text.append(f"Error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrackApp()
    window.show()
    sys.exit(app.exec_())
