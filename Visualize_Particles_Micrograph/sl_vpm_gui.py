'''
schroderlab-collection: Toolcollection for the processing of Cryo-EM Dataset
Copyright (C) 2025 Janus Lammert

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mrcfile
import starfile
import random
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage import exposure
from matplotlib.patches import FancyArrowPatch, Rectangle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QListWidget, QSplitter,
                             QGroupBox, QFormLayout, QProgressBar, QTabWidget, QTextEdit, QMessageBox, QFrame,
                             QSlider, QGridLayout, QToolButton, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
import json
import matplotlib.colors as mcolors
import matplotlib.patheffects as pae
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches

# --- Utility Functions ---
def read_starfile(starfile_path):
    try:
        star = starfile.read(starfile_path)
        particles = pd.DataFrame.from_dict(star["particles"])
        micrograph_particles = {}
        
        # Check if class assignment exists
        has_class = any(col for col in particles.columns if col.startswith("rlnClass"))
        class_col = next((col for col in particles.columns if col.startswith("rlnClass")), None)
        
        for _, row in particles.iterrows():
            name = os.path.basename(row["rlnMicrographName"])
            coord = (row["rlnCoordinateX"], row["rlnCoordinateY"])
            angles = {k: row[k] for k in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"] if k in row}
            
            # Add class assignment if available
            class_info = None
            if class_col:
                class_info = int(row[class_col])
                
            particle_info = {
                "coord": coord, 
                "angles": angles,
                "class": class_info
            }
            
            micrograph_particles.setdefault(name, []).append(particle_info)
        
        return micrograph_particles, has_class
    except Exception as e:
        QMessageBox.critical(None, "STAR File Error", f"Error reading STAR file:\n{e}")
        return {}, False

def process_image(img, lowpass=None, histeq=False):
    if lowpass:
        img = gaussian_filter(img, sigma=lowpass)
    if histeq:
        img = exposure.equalize_hist(img)
        img = (img * 255).astype(np.uint8)
    return img

def add_manual_scalebar(ax, pixel_size_angstrom, units="nm", target_length_nm=30, location='lower right', bar_height_fraction=0.015, color='white', fontsize=12, fontweight='bold', pad_x=0.02, pad_y=0.02, text_offset_y_factor=1.5):
    """Fügt eine manuelle Scalebar mit spezifischer Länge zu den Achsen hinzu (achsenrelative Positionierung)."""
    if not pixel_size_angstrom or pixel_size_angstrom <= 0:
        return

    # Umrechnung von Angström in die gewünschte Einheit (Standardmäßig nm)
    if units == "nm":
        pixel_size = pixel_size_angstrom / 10
    elif units == "A" or units == "Angstrom":
        pixel_size = pixel_size_angstrom
    else:
        raise ValueError(f"Ungültige Einheit: {units}. Unterstützte Einheiten sind 'nm' und 'Angstrom'.")

    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Berechne die Länge der Scalebar in Pixeln
    bar_length_pixels = target_length_nm / pixel_size

    # Berechne die relative Länge der Scalebar bezogen auf die x-Achsenrange
    bar_length_relative = bar_length_pixels / x_range

    # Calculate position in axis-relative coordinates
    if location == 'lower right':
        bar_x_relative = 1 - bar_length_relative - pad_x
        bar_y_relative = pad_y
        text_x_relative = 1 - bar_length_relative / 2 - pad_x
        text_y_relative = pad_y + bar_height_fraction * y_range * text_offset_y_factor / y_range
        ha = 'center'
        va = 'bottom'
    elif location == 'lower left':
        bar_x_relative = pad_x
        bar_y_relative = pad_y
        text_x_relative = pad_x + bar_length_relative / 2
        text_y_relative = pad_y + bar_height_fraction * y_range * text_offset_y_factor / y_range
        ha = 'center'
        va = 'bottom'
    elif location == 'upper right':
        bar_x_relative = 1 - bar_length_relative - pad_x
        bar_y_relative = 1 - pad_y - bar_height_fraction * y_range
        text_x_relative = 1 - bar_length_relative / 2 - pad_x
        text_y_relative = 1 - pad_y + (fontsize / (72 * y_range)) * text_offset_y_factor
        ha = 'center'
        va = 'top'
    elif location == 'upper left':
        bar_x_relative = pad_x
        bar_y_relative = 1 - pad_y - bar_height_fraction * y_range
        text_x_relative = pad_x + bar_length_relative / 2
        text_y_relative = 1 - pad_y + (fontsize / (72 * y_range)) * text_offset_y_factor
        ha = 'center'
        va = 'top'
    else:
        raise ValueError(f"Invalid location: {location}")

    # Convert relative positions to data coordinates
    bar_x = xlim[0] + bar_x_relative * x_range
    bar_y = ylim[0] + bar_y_relative * y_range
    bar_height = bar_height_fraction * y_range

    text_x_data = xlim[0] + text_x_relative * x_range
    text_y_data = ylim[0] + text_y_relative * y_range

    # Create the rectangle for the scale bar
    bar = patches.Rectangle((bar_x, bar_y), bar_length_pixels, bar_height,
                            linewidth=1, edgecolor=color, facecolor=color)
    ax.add_patch(bar)

    # Add the text label
    ax.text(text_x_data, text_y_data, f"{target_length_nm} {units}", ha=ha, va=va,
            color=color, fontsize=fontsize, fontweight=fontweight,
            bbox=None) # Entferne den schwarzen Hintergrund

def _round_to_nice(value):
    """Rundet einen Wert auf eine 'schöne' Zahl für die Scalebar."""
    magnitudes = [1, 2, 5, 10]
    exponent = np.floor(np.log10(value))
    mantissa = value / (10**exponent)
    for mag in magnitudes:
        if mantissa <= mag:
            return mag * (10**exponent)
    return magnitudes[-1] * (10**exponent)

             
def plot_micrograph(mrc_path, coordinates_dict, color_map, ax, lowpass=None, histeq=False,
                    show_arrows=False, arrow_angle="rlnAnglePsi", arrow_length=140,
                    arrow_width=2, point_size=10, pixel_size=None, color_by_class=False,
                    class_map=None):
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            img = mrc.data
        img = process_image(img, lowpass, histeq)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        legend_handles = []
        legend_labels = []
        
        for name, particles in coordinates_dict.items():
            if not particles:
                continue
                
            if color_by_class and class_map:
                # Group particles by class
                class_groups = {}
                for p in particles:
                    class_id = p.get("class", 0)  # Default to class 0 if not specified
                    if class_id not in class_groups:
                        class_groups[class_id] = []
                    class_groups[class_id].append(p)
                
                # Plot each class with its own color
                for class_id, class_particles in class_groups.items():
                    coords = [p["coord"] for p in class_particles]
                    x, y = zip(*coords)
                    color = class_map.get(class_id, 'white')  # Get color for this class
                    scatter = ax.scatter(x, y, s=point_size, color=color, edgecolors='black')
                    legend_labels.append(f"Class {class_id} ({len(class_particles)})")
                    legend_handles.append(scatter)
                    
                    if show_arrows and class_particles[0]["angles"] and arrow_angle in class_particles[0]["angles"]:
                        for p in class_particles:
                            angle_rad = np.radians(p["angles"].get(arrow_angle, 0))
                            dx = arrow_length * np.cos(angle_rad)
                            dy = arrow_length * np.sin(angle_rad)
                            ax.add_patch(FancyArrowPatch((p["coord"][0], p["coord"][1]),
                                                        (p["coord"][0] + dx, p["coord"][1] + dy),
                                                        color=color, linewidth=arrow_width, arrowstyle='-|>', alpha=0.8))
            else:
                # Color by star file (legacy behavior)
                coords = [p["coord"] for p in particles]
                x, y = zip(*coords)
                scatter = ax.scatter(x, y, s=point_size, color=color_map[name], edgecolors='black')
                legend_labels.append(f"{name} ({len(particles)})")
                legend_handles.append(scatter)
                
                if show_arrows and particles[0]["angles"] and arrow_angle in particles[0]["angles"]:
                    for p in particles:
                        angle_rad = np.radians(p["angles"].get(arrow_angle, 0))
                        dx = arrow_length * np.cos(angle_rad)
                        dy = arrow_length * np.sin(angle_rad)
                        ax.add_patch(FancyArrowPatch((p["coord"][0], p["coord"][1]),
                                                    (p["coord"][0] + dx, p["coord"][1] + dy),
                                                    color=color_map[name], linewidth=arrow_width, arrowstyle='-|>', alpha=0.8))
        
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper right')
            
        if pixel_size and pixel_size > 0:
            pixel_size_nm = pixel_size / 10  # Umrechnung von Angström in Nanometer (angenommen)
            units = "nm"  # Oder die passende Einheit für Ihre Daten
            add_manual_scalebar(ax, pixel_size, units=units, target_length_nm=30, location='lower right', color='white', fontsize=10, pad_x=0.01, pad_y=0.01, text_offset_y_factor=1.6)
            
    except Exception as e:
        QMessageBox.critical(None, "Error Plotting", f"Error loading MRC file:\n{mrc_path}\n\n{e}")

def calculate_statistics(micrograph_particles, color_by_class=False):
    if color_by_class:
        # Count particles by class
        class_counts = {}
        total_particles = 0
        
        for name, particles in micrograph_particles.items():
            for p in particles:
                class_id = p.get("class", 0)
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
                total_particles += 1
                
        text = f"Total Particles: {total_particles}\n"
        if class_counts:
            text += "Distribution by Class:\n"
            for class_id, count in sorted(class_counts.items()):
                text += f"  - Class {class_id}: {count} ({count/total_particles:.2%})\n"
    else:
        # Count particles by star file
        total_particles = sum(len(coords) for coords in micrograph_particles.values())
        ratios = {name: len(coords) / total_particles for name, coords in micrograph_particles.items()} if total_particles else {}
        avg_particles = total_particles / len(micrograph_particles) if micrograph_particles else 0
        
        text = f"Total Particles: {total_particles}\n"
        text += f"Average per STAR file: {avg_particles:.2f}\n"
        text += "Distribution by File:\n"
        for name, ratio in ratios.items():
            text += f"  - {name}: {len(micrograph_particles[name])} ({ratio:.2%})\n"
            
    return text

def generate_histogram(particles_by_star):
    hist_data = {"rlnAngleRot": [], "rlnAngleTilt": [], "rlnAnglePsi": []}
    for group in particles_by_star.values():
        for p in group:
            for key in hist_data:
                if key in p["angles"]:
                    hist_data[key].append(p["angles"][key])
    return hist_data

def generate_class_colors(num_classes):
    """Generate a nice color map for classes"""
    # Use tab10 colormap for up to 10 classes
    if num_classes <= 10:
        cmap = plt.cm.get_cmap('tab10', num_classes)
        return {i: cmap(i) for i in range(num_classes)}
    
    # For more classes, use hsv colormap
    cmap = plt.cm.get_cmap('hsv', num_classes)
    return {i: cmap(i) for i in range(num_classes)}

# --- Custom Widgets ---
class BetterButton(QPushButton):
    def __init__(self, text, parent=None, icon=None):
        super().__init__(text, parent)
        if icon:
            self.setIcon(QIcon(icon))
        self.setMinimumHeight(32)
        self.setStyleSheet("""
            QPushButton {
                background-color: #003D64;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)

class SliderWithValue(QWidget):
    valueChanged = pyqtSignal(float)
    
    def __init__(self, label, min_val, max_val, step, default, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.value_display = QLabel(f"{default:.1f}")
        
        # Compute slider range based on step
        self.step = step
        self.decimals = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        self.slider_scale = int(1 / step) if step < 1 else 1
        
        self.slider.setRange(int(min_val * self.slider_scale), int(max_val * self.slider_scale))
        self.slider.setValue(int(default * self.slider_scale))
        self.slider.setTickPosition(QSlider.TicksBelow)
        
        self.layout.addWidget(self.label, 1)
        self.layout.addWidget(self.slider, 4)
        self.layout.addWidget(self.value_display, 1)
        
        self.slider.valueChanged.connect(self._handle_slider_change)
        
    def _handle_slider_change(self, value):
        actual_value = value / self.slider_scale
        self.value_display.setText(f"{actual_value:.{self.decimals}f}")
        self.valueChanged.emit(actual_value)
        
    def value(self):
        return self.slider.value() / self.slider_scale
    
    def setValue(self, value):
        self.slider.setValue(int(value * self.slider_scale))

# --- GUI ---
class MPMGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPM Viewer")
        self.setGeometry(100, 100, 1400, 900)
        self.index = 0
        self.lowpass = 2.9
        self.histeq = False
        self.show_arrows = False
        self.arrow_angle = "rlnAnglePsi"
        self.arrow_length = 140
        self.arrow_width = 2
        self.point_size = 10
        self.pixel_size = 0.82
        self.micrographs = []
        self.mrc_folder = ""
        self.star_data = {}
        self.color_map = {}
        self.class_map = {}
        self.has_classes = False
        self.color_by_class = False
        self.init_ui()
        
        # Set a modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #2c3e50;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 2px;
                min-height: 24px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def save_settings(self):
        settings = {
            "lowpass": self.lowpass,
            "histeq": self.histeq,
            "show_arrows": self.show_arrows,
            "arrow_angle": self.arrow_angle,
            "arrow_length": self.arrow_length,
            "arrow_width": self.arrow_width,
            "point_size": self.point_size,
            "pixel_size": self.pixel_size,
            "mrc_folder": self.mrc_folder,
            "star_files": list(self.star_data.keys())
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", "settings.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(settings, f, indent=2)
                
            self.statusBar.showMessage(f"Settings saved to {path}", 3000)

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", "settings.json", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f:
                settings = json.load(f)
            self.lowpass = settings.get("lowpass", 2.9)
            self.histeq = settings.get("histeq", False)
            self.show_arrows = settings.get("show_arrows", False)
            self.arrow_angle = settings.get("arrow_angle", "rlnAnglePsi")
            self.arrow_length = settings.get("arrow_length", 140)
            self.arrow_width = settings.get("arrow_width", 2)
            self.point_size = settings.get("point_size", 10)
            self.pixel_size = settings.get("pixel_size", 0.82)

            # Apply to widgets
            self.lowpass_slider.setValue(self.lowpass)
            self.histeq_check.setChecked(self.histeq)
            self.arrows_check.setChecked(self.show_arrows)
            self.arrow_angle_combo.setCurrentText(self.arrow_angle)
            self.arrow_length_spin.setValue(self.arrow_length)
            self.arrow_width_spin.setValue(self.arrow_width)
            self.point_size_spin.setValue(self.point_size)
            self.pixel_size_spin.setValue(self.pixel_size or 0.0)
            
            self.statusBar.showMessage(f"Settings loaded from {path}", 3000)
            
    def refresh_display(self):
        if hasattr(self, 'last_file_displayed'):
            mrc_path = os.path.join(self.mrc_folder, self.last_file_displayed)
            coord_data = {name: data[self.last_file_displayed] for name, data in self.star_data.items() 
                         if self.last_file_displayed in data}
            plot_micrograph(mrc_path, coord_data, self.color_map, self.ax,
                           lowpass=self.lowpass, histeq=self.histeq,
                           show_arrows=self.show_arrows, arrow_angle=self.arrow_angle,
                           arrow_length=self.arrow_length, arrow_width=self.arrow_width,
                           point_size=self.point_size, pixel_size=self.pixel_size,
                           color_by_class=self.color_by_class, class_map=self.class_map)
            self.canvas.draw()


    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QHBoxLayout(self.main_widget)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        control_panel = QVBoxLayout(left_panel)
        control_panel.setSpacing(10)
        
        # File controls group
        file_group = QGroupBox("File Management")
        file_layout = QVBoxLayout(file_group)
        
        self.load_mrc_btn = BetterButton("Open MRC Folder")
        self.load_mrc_btn.clicked.connect(self.load_mrc_folder)
        file_layout.addWidget(self.load_mrc_btn)
        
        self.load_star_btn = BetterButton("Load STAR Files")
        self.load_star_btn.clicked.connect(self.load_star_files)
        file_layout.addWidget(self.load_star_btn)
        
        self.path_display = QLabel("No folders or files selected")
        self.path_display.setWordWrap(True)
        self.path_display.setStyleSheet("font-style: italic; color: #7f8c8d;")
        file_layout.addWidget(self.path_display)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = BetterButton("Previous")
        self.prev_btn.clicked.connect(self.previous_micrograph)
        
        self.next_btn = BetterButton("Next")
        self.next_btn.clicked.connect(self.next_micrograph)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        file_layout.addLayout(nav_layout)
        
        control_panel.addWidget(file_group)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        # Use custom slider instead of spin box for lowpass
        self.lowpass_slider = SliderWithValue("Lowpass:", 0, 20, 0.1, 2.9)
#        self.lowpass_slider.valueChanged.connect(lambda v: setattr(self, 'lowpass', v))
        self.lowpass_slider.valueChanged.connect(lambda v: [setattr(self, 'lowpass', v), self.refresh_display()])
        display_layout.addWidget(self.lowpass_slider)

        self.histeq_check = QCheckBox("Histogram Equalization")
#        self.histeq_check.stateChanged.connect(lambda s: setattr(self, 'histeq', bool(s)))
        self.histeq_check.stateChanged.connect(lambda s: [setattr(self, 'histeq', bool(s)), self.refresh_display()])
        display_layout.addWidget(self.histeq_check)
        
        # Marker settings subgroup
        marker_group = QGroupBox("Marker Settings")
        marker_layout = QFormLayout(marker_group)
        
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 50)
        self.point_size_spin.setValue(self.point_size)
        self.point_size_spin.valueChanged.connect(lambda v: [setattr(self, 'point_size', v), self.refresh_display()])
        marker_layout.addRow("Marker Size:", self.point_size_spin)
        
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.0, 100.0)
        self.pixel_size_spin.setSingleStep(0.1)
        self.pixel_size_spin.setValue(0.82)
        self.pixel_size_spin.setPrefix("Pixel Size (Å): ")
        self.pixel_size_spin.valueChanged.connect(lambda v: [setattr(self, 'pixel_size', v if v > 0 else None), self.refresh_display()])
        marker_layout.addRow("", self.pixel_size_spin)
        
        display_layout.addWidget(marker_group)
        
        # Arrow settings subgroup
        arrow_group = QGroupBox("Orientation Arrows")
        arrow_layout = QVBoxLayout(arrow_group)
        
        self.arrows_check = QCheckBox("Show Orientation Arrows")
        self.arrows_check.stateChanged.connect(lambda s: [setattr(self, 'show_arrows', bool(s)), self.refresh_display()])
        arrow_layout.addWidget(self.arrows_check)

        arrow_settings = QFormLayout()
        
        self.arrow_angle_combo = QComboBox()
        self.arrow_angle_combo.addItems(["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"])
        self.arrow_angle_combo.currentTextChanged.connect(lambda t: [setattr(self, 'arrow_angle', t), self.refresh_display()])
        arrow_settings.addRow("Angle:", self.arrow_angle_combo)

        self.arrow_length_spin = QSpinBox()
        self.arrow_length_spin.setRange(10, 500)
        self.arrow_length_spin.setValue(self.arrow_length)
        self.arrow_length_spin.valueChanged.connect(lambda v: [setattr(self, 'arrow_length', v), self.refresh_display()])
        arrow_settings.addRow("Length:", self.arrow_length_spin)

        self.arrow_width_spin = QDoubleSpinBox()
        self.arrow_width_spin.setRange(0.5, 10)
        self.arrow_width_spin.setSingleStep(0.5)
        self.arrow_width_spin.setValue(self.arrow_width)
        self.arrow_width_spin.valueChanged.connect(lambda v: [setattr(self, 'arrow_width', v), self.refresh_display()])
        arrow_settings.addRow("Width:", self.arrow_width_spin)
        
        arrow_layout.addLayout(arrow_settings)
        display_layout.addWidget(arrow_group)
        
        control_panel.addWidget(display_group)

        # Statistics and Info Panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        control_panel.addWidget(stats_group)

        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.save_btn = BetterButton("Save Image")
        self.save_btn.clicked.connect(self.save_current_image)
        buttons_layout.addWidget(self.save_btn)
        
        settings_layout = QHBoxLayout()
        self.save_settings_btn = BetterButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        settings_layout.addWidget(self.save_settings_btn)

        self.load_settings_btn = BetterButton("Load Settings")
        self.load_settings_btn.clicked.connect(self.load_settings)
        settings_layout.addWidget(self.load_settings_btn)
        
        control_panel.addLayout(buttons_layout)
        control_panel.addLayout(settings_layout)
        
        # Plot canvas (right side)
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(10, 10), dpi=100, facecolor='#f8f9fa')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        canvas_layout.addWidget(self.canvas)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(canvas_container, 1)  # Canvas takes more space

    def load_mrc_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose MRC Folder")
        if folder:
            self.mrc_folder = folder
            self.micrographs = sorted([f for f in os.listdir(self.mrc_folder) if f.endswith(".mrc")])
            self.index = 0
            self.update_path_display()
            
            self.statusBar.showMessage(f"Loaded {len(self.micrographs)} MRC files from {folder}", 3000)

    def load_star_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose STAR Files", filter="*.star")
        if not files:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one valid STAR file.")
            return
            
        self.star_data = {}
        self.color_map = {}
        self.class_map = {}
        
        # Set up coloring by STAR file (for multiple files)
        if len(files) > 1:
            # Standard colors for multiple star files
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']
            for i, f in enumerate(files):
                name = os.path.basename(f)
                particles_dict, has_classes = read_starfile(f)
                self.star_data[name] = particles_dict
                self.color_map[name] = colors[i % len(colors)]
                
            self.color_by_class = False
        else:
            # For single file, check if we have classes
            name = os.path.basename(files[0])
            particles_dict, has_classes = read_starfile(files[0])
            self.star_data[name] = particles_dict
            self.has_classes = has_classes
            
            # Set color by class if classes exist
            if has_classes:
                # Find all unique classes
                unique_classes = set()
                for micrograph_name, particles in particles_dict.items():
                    for p in particles:
                        if p.get("class") is not None:
                            unique_classes.add(p["class"])
                
                # Generate colors for all classes
                self.class_map = generate_class_colors(len(unique_classes))
                self.color_by_class = True
                
                self.statusBar.showMessage(f"Loaded file with {len(unique_classes)} classes", 3000)
            else:
                # No class information, use a single color
                self.color_map[name] = '#3498db'
                self.color_by_class = False
                
                self.statusBar.showMessage("Loaded STAR file with no class information", 3000)
        
        self.update_path_display()

    def save_current_image(self):
        if not hasattr(self, 'last_file_displayed'):
            QMessageBox.information(self, "No Image", "No image has been displayed yet.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", 
                                                f"{self.last_file_displayed}_annotated.png", 
                                                "PNG Image (*.png)")
        if save_path:
            self.figure.savefig(save_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Saved", f"Image saved to:{save_path}")
            self.statusBar.showMessage(f"Image saved to {save_path}", 3000)

    def update_path_display(self):
        msg = f"MRC Folder: {self.mrc_folder}\nSTAR Files: {', '.join(self.star_data.keys())}"
        if self.color_by_class:
            msg += "\nColoring by class assignment"
        self.path_display.setText(msg)

    def previous_micrograph(self):
        if not self.micrographs:
            return
        self.index = (self.index - 2) % len(self.micrographs)
        self.next_micrograph()

    def next_micrograph(self):
        if not self.micrographs:
            QMessageBox.warning(self, "No MRC Files", "Please select a valid MRC folder first.")
            return
        file = self.micrographs[self.index % len(self.micrographs)]
        self.index += 1
        mrc_path = os.path.join(self.mrc_folder, file)

        # Try to load coordinates (even if empty)
        coord_data = {name: data[file] for name, data in self.star_data.items() if file in data}
    
        # Update status
        self.statusBar.showMessage(f"Loading {file}...", 1000)

        # Plot with appropriate coloring
        plot_micrograph(mrc_path, coord_data, self.color_map, self.ax,
                        lowpass=self.lowpass, histeq=self.histeq,
                        show_arrows=self.show_arrows, arrow_angle=self.arrow_angle,
                        arrow_length=self.arrow_length, arrow_width=self.arrow_width,
                        point_size=self.point_size, pixel_size=self.pixel_size,
                        color_by_class=self.color_by_class, class_map=self.class_map)
        self.canvas.draw()

        # Statistik mit der bereits implementierten calculate_statistics Funktion berechnen
        stats = f"File: {file}\n"
        stats += calculate_statistics(coord_data, self.color_by_class)
        self.stats_text.setText(stats)

        self.last_file_displayed = file

# Main Block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MPMGui()
    viewer.show()
    sys.exit(app.exec_())
