#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full pipeline to automate Pupil Labs data processing and generate a saliency map.

This script orchestrates three main stages in a background thread while providing
an interactive visualization "waiting room" for the user.

1.  Automates the Pupil Player export process for the latest recording.
2.  Maps the exported gaze data from the world video onto a static reference image.
3.  Generates and displays a saliency map based on the mapped gaze data.

The interactive "waiting room" (from pipeline_pupil_inter.py) provides
multiple modes to explore saliency concepts while the pipeline runs.
"""

# --- General Imports ---
import os
import sys
import subprocess
import time
import configparser
import signal
import shutil
import argparse
import select
from collections import defaultdict
import threading
import glob

# --- Scientific and Image Processing Imports ---
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import gaussian_kde

# ==============================================================================
# SECTION 1: PUPIL LABS EXPORT AUTOMATION
# (This section remains unchanged from pipeline_pupil_full.py)
# ==============================================================================
def run_pupil_pipeline(shared_state):
    """
    Finds the latest recording, launches Pupil Player, triggers an export,
    and returns the path to the exported data directory.
    Updates a shared_state dictionary with its progress.
    """
    shared_state['status'] = "Stage 1: Starting Pupil Player Export..."
    print("\n" + "="*60)
    print("🚀 STAGE 1: AUTOMATING PUPIL PLAYER EXPORT")
    print("="*60)
    
    # --- 1. Load Configuration from config.ini ---
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        paths = config['Paths']
        pupil_project_dir = os.path.expanduser(paths['pupil_project_dir'])
        base_recordings_dir = os.path.join(pupil_project_dir, 'recordings')
        pupil_src_dir = os.path.join(pupil_project_dir, 'pupil_src')
        venv_activate_path = os.path.expanduser(paths['venv_activate_path'])
    except (KeyError, configparser.NoSectionError) as e:
        msg = f"❌ Error: 'config.ini' is missing a section or key: {e}"
        print(msg)
        shared_state['error'] = msg
        return None

    # --- 2. Determine Today's Recording Directory ---
    today_str = time.strftime('%Y_%m_%d')
    daily_folder_path = os.path.join(base_recordings_dir, today_str)
    
    shared_state['status'] = f"Stage 1: Checking for recordings in {today_str}..."
    print(f"2. ▶️  Checking for recordings in: {daily_folder_path}")
    if not os.path.isdir(daily_folder_path):
        msg = f"❌ Error: Today's recording directory not found."
        print(msg)
        shared_state['error'] = msg
        return None

    # --- 3. Identify the Latest Recording ---
    try:
        numeric_dirs = [d for d in os.listdir(daily_folder_path) if os.path.isdir(os.path.join(daily_folder_path, d)) and d.isdigit()]
        if not numeric_dirs: raise FileNotFoundError
        latest_recording_subdir = max(numeric_dirs, key=int)
        latest_recording_dir = os.path.join(daily_folder_path, latest_recording_subdir)
    except (OSError, FileNotFoundError):
        msg = f"❌ Error: No valid recordings found in '{daily_folder_path}'."
        print(msg)
        shared_state['error'] = msg
        return None
    print(f"3. ✅ Found latest recording: {latest_recording_dir}")

    # --- 4. Snapshot Existing Exports ---
    export_parent_dir = os.path.join(latest_recording_dir, "exports")
    existing_exports = set()
    if os.path.isdir(export_parent_dir):
        existing_exports = {d for d in os.listdir(export_parent_dir) if os.path.isdir(os.path.join(export_parent_dir, d)) and d.isdigit()}
    print(f"4. ℹ️  Found existing exports: {sorted(list(existing_exports)) if existing_exports else 'None'}")

    # --- 5. Launch Pupil Player ---
    pupil_player_command = (
        f"source {venv_activate_path} && "
        f"cd {pupil_src_dir} && "
        f"python -u main.py player {latest_recording_dir}"
    )
    shared_state['status'] = "Stage 1: Launching Pupil Player..."
    print("5. ⏳ Launching Pupil Player... (this may take a moment)")
    pupil_process = subprocess.Popen(
        pupil_player_command, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, preexec_fn=os.setsid
    )
    
    # --- 6. Wait for Initial Processing to Complete ---
    shared_state['status'] = "Stage 1: Waiting for fixation detection..."
    print("6. 👀 Waiting for fixation detection signal (timeout: 5s of inactivity)...")

    processing_complete = False
    any_ready_signal_seen = False

    while True:
        # Check if the user quit from the waiting room
        if shared_state.get('status') == 'stopped':
            print("   -> User requested stop. Terminating Pupil Player.")
            os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
            return None

        ready_to_read, _, _ = select.select([pupil_process.stdout], [], [], 5.0)
        if not ready_to_read:
            if any_ready_signal_seen: print("   -> ✅ Timeout reached after last output. Assuming player is ready.")
            else: print("   -> ⚠️  Timeout reached with no initial output. Assuming player is ready anyway.")
            processing_complete = True
            break
        line = pupil_process.stdout.readline()
        if not line:
            print("   -> Pupil Player process stream ended unexpectedly.")
            processing_complete = any_ready_signal_seen
            break
        stripped_line = line.strip()
        if stripped_line: print(f"   | {stripped_line}")
        if "World Video Exporter has been launched." in line: any_ready_signal_seen = True
        if "Starting fixation detection using 3d gaze data" in line:
            print("   -> ✅ Found fixation detection signal. Continuing immediately.")
            time.sleep(1)
            any_ready_signal_seen = True
            processing_complete = True
            break

    if not processing_complete:
        msg = "\n❌ Error: Player did not initialize correctly or closed before it was ready."
        print(msg)
        shared_state['error'] = msg
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None
    print("6. ✅ Initial processing complete. Player is ready for export command.")

    # --- 7. Trigger the Export ---
    shared_state['status'] = "Stage 1: Triggering export..."
    print("7. ⏳ Waiting for Pupil Player window to appear...")
    window_name_pattern = "Pupil Player:"
    window_found = False
    max_wait_seconds = 30
    for i in range(max_wait_seconds * 2):
        if shared_state.get('status') == 'stopped':
            print("   -> User requested stop. Terminating Pupil Player.")
            os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
            return None
            
        check_window_command = f'xdotool search --onlyvisible --name "{window_name_pattern}"'
        result = subprocess.run(check_window_command, shell=True, capture_output=True, executable='/bin/bash')
        if result.returncode == 0:
            print(f"7.1 ✅ Found Pupil Player window matching '{window_name_pattern}'.")
            window_found = True
            break
        time.sleep(0.5)

    if not window_found:
        msg = f"   ❌ Error: Pupil Player window did not appear within {max_wait_seconds} seconds."
        print(msg)
        shared_state['error'] = msg
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None

    print("7.2 ⌨️ Simulating 'e' key press to trigger export...")
    try:
        time.sleep(1) 
        keypress_command = f'xdotool search --onlyvisible --name "{window_name_pattern}" windowactivate --sync key e'
        subprocess.run(keypress_command, shell=True, check=True, capture_output=True, text=True, executable='/bin/bash')
        print("   ✅ Keypress 'e' sent successfully.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        msg = f"   ❌ Error: Failed to send keypress. Is 'xdotool' installed? Reason: {e}"
        print(msg)
        shared_state['error'] = msg
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None

    # --- 8. Wait for Export to Finish ---
    shared_state['status'] = "Stage 1: Finalizing export..."
    print("8. 👀 Monitoring for 'Export done' confirmation...")
    export_confirmed = False
    for line in pupil_process.stdout:
        if shared_state.get('status') == 'stopped':
            print("   -> User requested stop. Terminating Pupil Player.")
            os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
            return None
        if "Export done:" in line:
            print("   ✅ Detected 'Export done' message!")
            export_confirmed = True
            break
            
    # --- 9. Find the New Export Folder ---
    final_export_path = None
    if export_confirmed:
        print("9. 🔎 Checking filesystem for the new export folder...")
        time.sleep(1)
        current_exports = {d for d in os.listdir(export_parent_dir) if os.path.isdir(os.path.join(export_parent_dir, d)) and d.isdigit()}
        newly_created = current_exports - existing_exports
        if newly_created:
            new_export_folder_name = max(newly_created, key=int)
            final_export_path = os.path.join(export_parent_dir, new_export_folder_name)
            print(f"   ✅ Found new export folder: {new_export_folder_name}")
        else: print("   ⚠️ Could not find a new export folder on the filesystem.")
    else:
        msg = "\n   ❌ Error: 'Export done' message was not detected."
        print(msg)
        shared_state['error'] = msg

    # --- 10. Close Pupil Player ---
    print("10. 🚮 Closing Pupil Player application...")
    try:
        pgid = os.getpgid(pupil_process.pid)
        os.killpg(pgid, signal.SIGKILL)
        print("   ✅ Pupil Player closed successfully.")
    except Exception as e:
        print(f"   ⚠️ An error occurred while closing Pupil Player: {e}")

    return os.path.abspath(final_export_path) if final_export_path else None

# ==============================================================================
# SECTION 2: GAZE MAPPING
# (This section remains unchanged from pipeline_pupil_full.py)
# ==============================================================================
def run_map_gaze(export_dir, ref_image_path, frame_step, shared_state):
    shared_state['status'] = "Stage 2: Starting Gaze Mapping..."
    print("\n" + "="*60)
    print("🖼️  STAGE 2: MAPPING GAZE TO REFERENCE IMAGE")
    print("="*60)

    gaze_csv = os.path.join(export_dir, 'gaze_positions.csv')
    fixations_csv = os.path.join(export_dir, 'fixations.csv')
    video_path = os.path.join(export_dir, 'world.mp4')
    out_dir = os.path.join(export_dir, 'mapGaze_output')
    
    for path in [gaze_csv, fixations_csv, video_path, ref_image_path]:
        if not os.path.exists(path):
            msg = f"❌ Error: Required file not found at {path}"
            print(msg)
            shared_state['error'] = msg
            return None
    
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(ref_image_path, out_dir)
    print(f"✅ Output will be saved to: {out_dir}")

    gaze_df = pd.read_csv(gaze_csv)
    fixations_df = pd.read_csv(fixations_csv)

    min_frame_index = gaze_df['world_index'].min()
    gaze_df['world_index'] -= min_frame_index
    fixations_df['start_frame_index'] -= min_frame_index
    fixations_df['end_frame_index'] -= min_frame_index

    print("⏳ Pre-processing gaze and fixation data for fast lookup...")
    gaze_by_frame = gaze_df.groupby('world_index').apply(lambda x: x.to_dict('records')).to_dict()
    fixations_by_frame = defaultdict(list)
    for _, row in fixations_df.iterrows():
        for frame_idx in range(int(row['start_frame_index']), int(row['end_frame_index']) + 1):
            fixations_by_frame[frame_idx].append(row.to_dict())
    print("✅ Data pre-processing complete.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_img = cv2.imread(ref_image_path)
    ref_gray, _ = preprocess_frame(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY))
    detector = cv2.AKAZE_create()
    ref_kp, ref_des = detector.detectAndCompute(ref_gray, None)

    idx, gaze_mapped_records, fixations_mapped_records = 0, [], []
    last_known_Hs = None
    
    print(f"⏳ Processing {total_frames} video frames (homography step = {frame_step})...")
    while True:
        # Check if the user quit from the waiting room
        if shared_state.get('status') == 'stopped':
            print("   -> User requested stop. Terminating gaze mapping.")
            cap.release()
            return None
            
        ret, frame = cap.read()
        if not ret: break

        if idx % frame_step == 0:
            Hs = process_single_frame(frame, ref_kp, ref_des, detector, ref_gray.shape)
            if Hs: last_known_Hs = Hs

        if last_known_Hs:
            frame_gaze_list = gaze_by_frame.get(idx, [])
            for row in frame_gaze_list:
                wx, wy = row['norm_pos_x'] * frame.shape[1], (1 - row['norm_pos_y']) * frame.shape[0]
                rx, ry = map_coords((wx, wy), last_known_Hs['H_frame2ref'])
                gaze_mapped_records.append({'worldFrame': idx, 'gaze_ts': row['gaze_timestamp'], 'confidence': row['confidence'], 'world_gazeX': wx, 'world_gazeY': wy, 'ref_gazeX': rx, 'ref_gazeY': ry})
            
            this_frame_fixations = fixations_by_frame.get(idx, [])
            for fix_row in this_frame_fixations:
                fix_wx, fix_wy = fix_row['norm_pos_x'] * frame.shape[1], (1 - fix_row['norm_pos_y']) * frame.shape[0]
                fix_rx, fix_ry = map_coords((fix_wx, fix_wy), last_known_Hs['H_frame2ref'])
                fixations_mapped_records.append({'fixation_id': fix_row['id'], 'start_ts': fix_row['start_timestamp'], 'duration': fix_row['duration'], 'confidence': fix_row['confidence'], 'worldFrame': idx, 'world_fixX': fix_wx, 'world_fixY': fix_wy, 'ref_fixX': fix_rx, 'ref_fixY': fix_ry})

        if idx % 100 == 0:
            status_msg = f"Stage 2: Mapping frame {idx}/{total_frames}..."
            shared_state['status'] = status_msg
            found_status = "Found" if last_known_Hs else "Not Found"
            print(f"   ... processed frame {idx}/{total_frames} (Homography: {found_status})")
        idx += 1

    cap.release()
    print("✅ Video processing complete.")

    if gaze_mapped_records:
        gaze_mapped_df = pd.DataFrame(gaze_mapped_records)
        gaze_output_path = os.path.join(out_dir, 'gazeData_mapped.tsv')
        gaze_mapped_df.to_csv(gaze_output_path, sep='\t', index=False)
        print(f"✅ Mapped gaze data saved to {gaze_output_path}")

    if fixations_mapped_records:
        fixations_mapped_df = pd.DataFrame(fixations_mapped_records).drop_duplicates(subset=['fixation_id', 'worldFrame'])
        fixations_output_path = os.path.join(out_dir, 'fixations_mapped.tsv')
        fixations_mapped_df.to_csv(fixations_output_path, sep='\t', index=False)
        print(f"✅ Mapped fixations data saved to {fixations_output_path}")
    
    return out_dir

def adjust_gamma(frame, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def local_brightness_norm(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def preprocess_frame(frame, simplify=True):
    if simplify: return frame, None # Simplified for speed during mapping
    frame = adjust_gamma(frame, gamma=1.5)
    frame = local_brightness_norm(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    return gray, None

def find_matches(kp1, des1, kp2, des2, dist_ratio=0.7, min_good_matches=12):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if des1 is None or des2 is None: return None, None
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < dist_ratio * n.distance:
                good.append(m)
    except cv2.error:
        return None, None
        
    if len(good) >= min_good_matches:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2
    return None, None

def compute_homography(pts_ref, pts_frame, ransac_thresh=8.0):
    H, mask = cv2.findHomography(pts_ref.reshape(-1,1,2), pts_frame.reshape(-1,1,2), cv2.RANSAC, ransac_thresh)
    return H, mask

def is_homography_reasonable(H, ref_shape, min_area=2000, max_aspect_ratio=4.0):
    h, w = ref_shape
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area = cv2.contourArea(projected.astype(np.float32))
    if area < min_area: return False
    side_lengths = [np.linalg.norm(projected[i] - projected[(i+1)%4]) for i in range(4)]
    ratio = max(side_lengths) / (min(side_lengths) + 1e-6)
    if ratio > max_aspect_ratio: return False
    return True

def process_single_frame(frame, ref_kp, ref_des, detector, ref_shape):
    gray, _ = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(kp) < 12: return None
    pts_ref, pts_frame = find_matches(ref_kp, ref_des, kp, des)
    if pts_ref is None: return None
    H_ref2frame, _ = compute_homography(pts_ref, pts_frame)
    if H_ref2frame is None or not is_homography_reasonable(H_ref2frame, ref_shape): return None
    _, H_frame2ref = cv2.invert(H_ref2frame)
    return {'H_ref2frame': H_ref2frame, 'H_frame2ref': H_frame2ref}

def map_coords(pt, H):
    arr = np.array(pt, dtype=np.float32).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(arr, H)
    x, y = dst.ravel()
    return float(x), float(y)


# ==============================================================================
# SECTION 3: SALIENCY MAP GENERATION
# (This section remains unchanged from pipeline_pupil_full.py)
# ==============================================================================
def create_saliency_map(jpg_file, gaze_data_file, output_path, shared_state, greyscale_output_path=None, method="gaussian", sigma=50, alpha=0.6):
    shared_state['status'] = "Stage 3: Generating Saliency Map..."
    print("\n" + "="*60)
    print("🔥 STAGE 3: GENERATING SALIENCY MAP")
    print("="*60)

    try:
        # Check for user stop request
        if shared_state.get('status') == 'stopped':
            print("   -> User requested stop. Skipping saliency map.")
            return False

        img = cv2.imread(jpg_file)
        if img is None:
            msg = f"❌ Error: Could not load background image from {jpg_file}"
            print(msg)
            shared_state['error'] = msg
            return False
        h, w = img.shape[:2]
        
        gaze_df = pd.read_csv(gaze_data_file, sep='\t')
        
        valid_gaze = gaze_df.dropna(subset=['ref_gazeX', 'ref_gazeY'])
        valid_gaze = valid_gaze[
            (valid_gaze['ref_gazeX'] >= 0) & (valid_gaze['ref_gazeX'] < w) &
            (valid_gaze['ref_gazeY'] >= 0) & (valid_gaze['ref_gazeY'] < h)
        ]
        
        if len(valid_gaze) == 0:
            print("⚠️ Warning: No valid gaze points found within the image boundaries.")
            cv2.imwrite(output_path, img)
            if greyscale_output_path:
                cv2.imwrite(greyscale_output_path, np.zeros((h, w), dtype=np.uint8))
            return True
        
        x_coords, y_coords = valid_gaze['ref_gazeX'].values, valid_gaze['ref_gazeY'].values
        
        print(f"⏳ Creating heatmap from {len(x_coords)} valid gaze points...")
        
        if method == "gaussian":
            heat_map_raw = create_gaussian_heatmap(x_coords, y_coords, w, h, sigma)
        else: # "kde"
            heat_map_raw = create_kde_heatmap(x_coords, y_coords, w, h)
        
        if heat_map_raw.max() == 0:
            print("⚠️ Warning: Heatmap is empty after processing points.")
            cv2.imwrite(output_path, img)
            if greyscale_output_path:
                cv2.imwrite(greyscale_output_path, np.zeros((h, w), dtype=np.uint8))
            return True

        heat_map_norm = (heat_map_raw / heat_map_raw.max() * 255).astype(np.uint8)
        
        if greyscale_output_path:
            cv2.imwrite(greyscale_output_path, heat_map_norm)
            print(f"✅ Greyscale saliency map saved to {greyscale_output_path}")

        heat_map_colored = cv2.applyColorMap(heat_map_norm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 1 - alpha, heat_map_colored, alpha, 0)
        
        cv2.imwrite(output_path, result)
        print(f"✅ Saliency map saved to {output_path}")
        return True

    except Exception as e:
        msg = f"❌ An error occurred during saliency map generation: {e}"
        print(msg)
        shared_state['error'] = msg
        return False

def create_gaussian_heatmap(x, y, w, h, sigma=50):
    m = np.zeros((h, w), dtype=np.float64)
    for i, j in zip(x, y):
        i, j = int(round(i)), int(round(j))
        if 0 <= i < w and 0 <= j < h: m[j, i] += 1
    return ndimage.gaussian_filter(m, sigma=sigma)

def create_kde_heatmap(x, y, w, h):
    grid_x = np.linspace(0, w - 1, w // 4); grid_y = np.linspace(0, h - 1, h // 4)
    xx, yy = np.meshgrid(grid_x, grid_y)
    positions = np.vstack([xx.ravel(), yy.ravel()]); values = np.vstack([x, y])
    kernel = gaussian_kde(values); density = kernel(positions).reshape(xx.shape)
    return cv2.resize(density, (w, h))


# =============================================================================
# SECTION 4: NEW INTERACTIVE WAITING ROOM
# (This class is from pipeline_pupil_inter.py)
# =============================================================================

class InteractiveWaitingRoom:
    """
    Manages an interactive OpenCV window while waiting for the main pipeline.

    Provides multiple modes for the user to explore saliency concepts.
    """
    def __init__(self, ref_image_path, model_pred_path, model_pred_greyscale_path):
        self.window_name = "Interactive Waiting Room (Pipeline Running...)"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- Load and prepare assets ---
        self.ref_image = self._load_image(ref_image_path, "Reference Image")
        self.model_pred = self._load_image(model_pred_path, "Model Prediction")
        self.model_pred_greyscale = self._load_image(model_pred_greyscale_path, "Model Prediction Greyscale", grayscale=True)

        if self.ref_image is None:
            self.ref_image = self._create_placeholder("Reference Image Not Found", (600, 800))
        
        h, w = self.ref_image.shape[:2]

        # --- Mode 1: Deform assets ---
        self.deformed_image = self.ref_image.copy()

        # --- Mode 2: Overlay assets ---
        if self.model_pred_greyscale is not None:
            self.saliency_overlay = cv2.applyColorMap(self.model_pred_greyscale, cv2.COLORMAP_JET)
        else:
            placeholder_gray = self._create_placeholder("Greyscale Saliency Not Found", (h,w), channels=1)
            self.saliency_overlay = cv2.applyColorMap(placeholder_gray, cv2.COLORMAP_JET)

        # --- Mode 3: Tunnel Vision assets ---
        self.blurred_image = cv2.GaussianBlur(self.ref_image, (101, 101), 0)


        # --- Mode 4: A/B Test assets ---
        h, w = self.ref_image.shape[:2] # Base dimensions from reference image
        w_new_model = w # Default to reference width

        if self.model_pred is not None:
             # --- FIX: Resize proportionally like the final 2x2 grid does ---
             h_model, w_model = self.model_pred.shape[:2]
             w_new_model = int(w_model * h / h_model) # Calculate proportional width
             self.model_pred_resized = cv2.resize(self.model_pred, (w_new_model, h))
             self.ab_composite_image = np.hstack((self.ref_image, self.model_pred_resized))
        else:
            # Create a placeholder matching the reference image size
            placeholder_pred = self._create_placeholder("Model Prediction Not Found", (h,w))
            w_new_model = placeholder_pred.shape[1]
            self.ab_composite_image = np.hstack((self.ref_image, placeholder_pred))
            # --- BUGFIX: Assign placeholder if original is missing ---
            self.model_pred_resized = placeholder_pred

        # --- NEW: Create the right-hand placeholder for the selection phase ---
        
        # --- FIX for cv2.error: Create a black image of the correct size ---
        black_image = np.zeros_like(self.model_pred_resized)
        
        # --- Blend the resized model with the black image ---
        self.ab_selection_placeholder = cv2.addWeighted(self.model_pred_resized, 0.05, black_image, 0.95, 0)
        cv2.putText(self.ab_selection_placeholder, "Guess 3 hotspots on the LEFT and see you SCORE", (30, h // 2), self.font, 1.2, (255, 255, 255), 3)



        self.ab_model_hotspots = [
        (300, 520) ,  # <-- Replace with your (x1, y1)
        (610, 330) ,  # <-- Replace with your (x2, y2)
        (778, 555)   # <-- Replace with your (x3, y3)
        ]


        # --- State variables for all modes ---
        self.mode = 'tunnel'
        self.dirty = True
        
        # Deform state
        self.deform_center = None
        self.deform_radius = 150
        self.zoom_strength = 0.5
        
        # Overlay state
        self.overlay_intensity = 50
        
        # Tunnel Vision state
        self.tunnel_center = (w // 2, h // 2)
        self.tunnel_radius = 120
        
        # A/B Test state
        self.ab_user_clicks = []
        self.ab_show_results = False

    def _load_image(self, path, name, grayscale=False):
        if not os.path.exists(path):
            print(f"❌ ERROR: Cannot find {name} at path: {path}")
            return None
        print(f"✅ Loaded {name} from {path}")
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)

    def _create_placeholder(self, text, size=(600, 800), channels=3):
        shape = (size[0], size[1], channels) if channels > 1 else (size[0], size[1])
        placeholder = np.zeros(shape, dtype=np.uint8)
        (text_width, text_height), _ = cv2.getTextSize(text, self.font, 1, 2)
        text_x = (placeholder.shape[1] - text_width) // 2
        text_y = (placeholder.shape[0] + text_height) // 2
        cv2.putText(placeholder, text, (text_x, text_y), self.font, 1, (255, 255, 255), 2)
        return placeholder

    def _mouse_callback(self, event, x, y, flags, param):
        if self.mode == 'deform' and event == cv2.EVENT_LBUTTONDOWN:
            self.deform_center = (x, y)
            self.dirty = True
        elif self.mode == 'tunnel' and event == cv2.EVENT_MOUSEMOVE:
            self.tunnel_center = (x, y)
            self.dirty = True


        elif self.mode == 'ab_test' and event == cv2.EVENT_LBUTTONUP:
            if not self.ab_show_results:
                try:
                    # --- FIX: Map window (x,y) to composite image (x,y) ---
                    _, _, w_win, h_win = cv2.getWindowImageRect(self.window_name)
                    h_comp, w_comp = self.ab_composite_image.shape[:2]

                    # Calculate the coordinate on the full-size composite image
                    img_x = int(x * w_comp / w_win)
                    img_y = int(y * h_comp / h_win)

                    # Only accept clicks on the left-hand image
                    if img_x < self.ref_image.shape[1]:
                        self.ab_user_clicks.append((img_x, img_y))
                        self.dirty = True
                        if len(self.ab_user_clicks) == 3:
                            self.ab_show_results = True
                    
                except (cv2.error, AttributeError, ValueError): 
                    # Fallback in case images aren't ready
                    print("Warning: Could not map click coordinates.")
                    pass



    def _intensity_slider_callback(self, val):
        if self.mode == 'overlay':
            self.overlay_intensity = val
            self.dirty = True

    def _draw_instructions(self, image, pipeline_status):
        cv2.rectangle(image, (0, 0), (image.shape[1], 95), (0,0,0), -1)
        cv2.putText(image, "Press 'q' to quit", (15, 30), self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Modes: '1' Tunnel, '2' Overlay, '3' Deform, '4' A/B Test", (15, 60), self.font, 0.7, (255, 255, 255), 2)
        
        # Display pipeline status
        status_text = f"Pipeline Status: {pipeline_status}"
        (text_width, _), _ = cv2.getTextSize(status_text, self.font, 0.7, 2)
        cv2.putText(image, status_text, (image.shape[1] - text_width - 15, 30), self.font, 0.7, (0, 200, 255), 2)

        if self.mode == 'deform':
            cv2.putText(image, "[Deform] Left-Click to expand. 'r' to reset.", (15, image.shape[0] - 20), self.font, 0.8, (0, 255, 255), 2)
        elif self.mode == 'overlay':
            cv2.putText(image, "[Overlay] Use slider to change saliency intensity", (15, image.shape[0] - 20), self.font, 0.8, (0, 255, 255), 2)
        elif self.mode == 'tunnel':
            cv2.putText(image, "[Tunnel Vision] Move mouse to reveal focus area", (15, image.shape[0] - 20), self.font, 0.8, (0, 255, 255), 2)
        elif self.mode == 'ab_test':
            if self.ab_show_results:
                cv2.putText(image, "[A/B Test] Press 'r' to try again", (15, image.shape[0] - 20), self.font, 0.8, (0, 255, 255), 2)
            else:
                clicks_left = 3 - len(self.ab_user_clicks)
                plural = "s" if clicks_left != 1 else ""
                cv2.putText(image, f"[A/B Test] Click to select {clicks_left} more hotspot{plural}", (15, image.shape[0] - 20), self.font, 0.8, (0, 255, 255), 2)
        return image
    

    def calculate_matched_score(self, user_clicks, model_hotspots, normalization_factor):
        """
        Calculates a score based on the user's proposed 1-to-1 greedy 
        matching algorithm. It repeatedly finds the closest user-model pair,
        records the distance, and removes them from the pool.
        """
        # Make copies to avoid modifying the original lists during calculation
        remaining_user_pts = list(user_clicks)
        remaining_model_pts = list(model_hotspots)
        
        matched_distances = []

        # Loop 3 times to find the 3 unique pairs
        for _ in range(len(model_hotspots)):
            if not remaining_user_pts or not remaining_model_pts:
                break # Stop if one of the lists is empty

            min_dist = float('inf')
            best_user_idx = -1
            best_model_idx = -1

            # Find the single closest pair among all remaining points
            for u_idx, u_pt in enumerate(remaining_user_pts):
                for m_idx, m_pt in enumerate(remaining_model_pts):
                    dist = np.linalg.norm(np.array(u_pt) - np.array(m_pt))
                    if dist < min_dist:
                        min_dist = dist
                        best_user_idx = u_idx
                        best_model_idx = m_idx
            
            # If a valid pair was found, "match" them
            if best_user_idx != -1:
                matched_distances.append(min_dist)
                # Remove the matched points so they can't be used again
                remaining_user_pts.pop(best_user_idx)
                remaining_model_pts.pop(best_model_idx)

        # Calculate the final score based on the sum of the unique matched distances
        total_matched_dist = sum(matched_distances)
        
        # Use a scoring formula similar to the original, normalized by image width
        score = max(0, int(100 - (total_matched_dist / normalization_factor) * 100))
        return score

    def run(self, shared_state):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.createTrackbar('Intensity', self.window_name, self.overlay_intensity, 100, self._intensity_slider_callback)
        cv2.setTrackbarPos('Intensity', self.window_name, 0)

        print("\n--- Starting Interactive Waiting Room ---")
        print("Modes: 1:Deform, 2:Overlay, 3:Tunnel Vision, 4:A/B Test")
        print("Pipeline is running in the background...")

        while shared_state.get('status', 'running') == 'running' or 'Stage' in shared_state.get('status', ''):
            current_status = shared_state.get('status', 'running')
            
            if self.dirty or 'Stage' in current_status: # Force redraw if status text changes
                display_image = None
                
                # --- Mode 1: Deform Logic ---
                if self.mode == 'deform':
                    if self.deform_center:
                        h, w = self.deformed_image.shape[:2]
                        cx, cy = self.deform_center
                        source_image = self.deformed_image
                        y_coords, x_coords = np.mgrid[0:h, 0:w]
                        dx, dy = x_coords - cx, y_coords - cy
                        distance = np.sqrt(dx**2 + dy**2)
                        sigma = self.deform_radius / 2.5
                        gaussian_blob = np.exp(-(distance**2) / (2 * sigma**2))
                        zoom_field = 1.0 + self.zoom_strength * gaussian_blob
                        new_x = cx + dx / zoom_field
                        new_y = cy + dy / zoom_field
                        self.deformed_image = cv2.remap(source_image, new_x.astype(np.float32), new_y.astype(np.float32), cv2.INTER_LINEAR)
                        self.deform_center = None
                    display_image = self.deformed_image.copy()

                # --- Mode 2: Overlay Logic ---
                elif self.mode == 'overlay':
                    alpha = self.overlay_intensity / 100.0
                    beta = 1.0 - alpha
                    display_image = cv2.addWeighted(self.ref_image, beta, self.saliency_overlay, alpha, 0.0)
                
                # --- Mode 3: Tunnel Vision Logic ---
                elif self.mode == 'tunnel':
                    display_image = self.blurred_image.copy()
                    mask = np.zeros(self.ref_image.shape, dtype=np.uint8)
                    cv2.circle(mask, self.tunnel_center, self.tunnel_radius, (255,255,255), -1)
                    clear_area = cv2.bitwise_and(self.ref_image, mask)
                    blurred_area = cv2.bitwise_and(display_image, cv2.bitwise_not(mask))
                    display_image = cv2.add(clear_area, blurred_area)


                # --- Mode 4: A/B Test Logic (NEW: CONSTANT SIZE) ---
                elif self.mode == 'ab_test':
                    
                    base_composite = None
                    
                    if self.ab_show_results:
                        # --- RESULTS PHASE ---
                        base_composite = self.ab_composite_image.copy()
                        w_offset = self.ref_image.shape[1] # Width of left image
                        
                        # Draw user clicks (blue)
                        for i, user_pt in enumerate(self.ab_user_clicks):
                            cv2.circle(base_composite, user_pt, 20, (255, 150, 0), 3)
                            cv2.putText(base_composite, str(i+1), (user_pt[0] - 10, user_pt[1] + 10), self.font, 0.8, (255, 255, 255), 2)
                        
                        # Draw model hotspots (red)
                        for i, model_pt in enumerate(self.ab_model_hotspots):
                            model_draw_pt = (model_pt[0] + w_offset, model_pt[1])
                            cv2.circle(base_composite, model_draw_pt, 20, (0, 0, 255), 3)
                            cv2.putText(base_composite, str(i+1), (model_draw_pt[0] - 10, model_draw_pt[1] + 10), self.font, 0.8, (255, 255, 255), 2)
                        
                        # --- NEW: Calculate score with 1-to-1 matching algorithm ---
                        # The normalization factor is a rough estimate of a "bad" total distance
                        normalization_factor = self.ref_image.shape[1] * 1.5 
                        score = self.calculate_matched_score(
                            self.ab_user_clicks, 
                            self.ab_model_hotspots, 
                            normalization_factor
                        )
                        # --- END of new calculation ---
                    
                    else:
                        # --- SELECTION PHASE ---
                        left_image = self.ref_image.copy()
                        # Draw clicks so far
                        for i, pt in enumerate(self.ab_user_clicks):
                            cv2.circle(left_image, pt, 20, (255, 150, 0), 3)
                            cv2.putText(left_image, str(i+1), (pt[0] - 10, pt[1] + 10), self.font, 0.8, (255, 255, 255), 2)
                        
                        # Stack with the new placeholder
                        base_composite = np.hstack((left_image, self.ab_selection_placeholder))

                    
                    # --- UNIFIED RESIZING (FOR BOTH PHASES) ---
                    MAX_DISPLAY_WIDTH = 1800
                    h, w = base_composite.shape[:2]
                    scale = 1.0 
                    
                    if w > MAX_DISPLAY_WIDTH:
                        scale = MAX_DISPLAY_WIDTH / w
                        h_scaled = int(h * scale)
                        display_image = cv2.resize(base_composite, (MAX_DISPLAY_WIDTH, h_scaled), interpolation=cv2.INTER_AREA)
                    else:
                        display_image = base_composite.copy()

                    # --- DRAW SCORE (only in results phase, but after scaling) ---
                    if self.ab_show_results:
                        score_x_pos = int((self.ref_image.shape[1] * scale / 2) - 80) 
                        cv2.putText(display_image, f"Score: {score}", (score_x_pos, 140), self.font, 1.5, (0,255,0), 3)


                if display_image is not None:
                    display_image = self._draw_instructions(display_image, current_status)
                    cv2.imshow(self.window_name, display_image)
                self.dirty = False

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                shared_state['status'] = 'stopped' # Signal the pipeline thread to stop
                break
            

            # --- Mode Switching ---
            new_mode = None
            if key == ord('1'): new_mode = 'tunnel'
            elif key == ord('2'): new_mode = 'overlay'
            elif key == ord('3'): new_mode = 'deform'
            elif key == ord('4'): new_mode = 'ab_test'
            
            if new_mode and self.mode != new_mode:
                print(f"Switching to {new_mode.replace('_', ' ').title()} Mode")
                self.mode = new_mode
                cv2.setTrackbarPos('Intensity', self.window_name, self.overlay_intensity if new_mode == 'overlay' else 0)
                self.dirty = True

            # --- Resetting ---
            if key == ord('r'):
                if self.mode == 'deform':
                    print("Resetting deformations.")
                    self.deformed_image = self.ref_image.copy()
                    self.dirty = True
                elif self.mode == 'ab_test':
                    print("Resetting A/B Test.")
                    self.ab_user_clicks = []
                    self.ab_show_results = False
                    self.dirty = True
        
        cv2.destroyAllWindows()
        print("--- Exiting Interactive Waiting Room ---")


# ==============================================================================
# SECTION 5: FULL BACKGROUND PIPELINE WORKER
# (This now runs all stages, 1, 2, and 3, in the background)
# ==============================================================================
def full_pipeline_worker(args, shared_state):
    """
    Runs the entire pipeline (Stages 1, 2, 3) in the background.
    Updates shared_state: 'status' ('running', 'complete', 'error'), 'error', etc.
    """
    try:
        # --- STAGE 1: PUPIL EXPORT ---
        export_dir = run_pupil_pipeline(shared_state)
        
        # Check if user stopped during stage 1
        if shared_state.get('status') == 'stopped':
            print("[Pipeline Thread] User stopped during Stage 1.")
            return

        if not export_dir:
            if not shared_state.get('error'):
                shared_state['error'] = "Stage 1 failed: Pupil export was unsuccessful."
            shared_state['status'] = 'error'
            return

        # --- STAGE 2: GAZE MAPPING ---
        mapgaze_dir = run_map_gaze(export_dir, args.ref_image, args.frame_step, shared_state)
        
        # Check if user stopped during stage 2
        if shared_state.get('status') == 'stopped':
            print("[Pipeline Thread] User stopped during Stage 2.")
            return

        if not mapgaze_dir:
            if not shared_state.get('error'):
                shared_state['error'] = "Stage 2 failed: Gaze mapping was unsuccessful."
            shared_state['status'] = 'error'
            return
        
        # --- STAGE 3: SALIENCY MAP ---
        gaze_mapped_file = os.path.join(mapgaze_dir, 'gazeData_mapped.tsv')
        fixations_mapped_file = os.path.join(mapgaze_dir, 'fixations_mapped.tsv')
        
        if os.path.exists(fixations_mapped_file):
            shared_state['fixations_mapped_path'] = fixations_mapped_file # Save for final display
        
        if not os.path.exists(gaze_mapped_file):
            print(f"⚠️ Mapped gaze file not found: {gaze_mapped_file}. Skipping saliency map.")
            shutil.copy(args.ref_image, args.output) # Provide original image as output
        else:
            # We need paths for the greyscale maps for the final 2x2 display
            greyscale_saliency_path = "images/saliency_map_greyscale.jpg" 
            success = create_saliency_map(
                args.ref_image, 
                gaze_mapped_file, 
                args.output, 
                shared_state, 
                greyscale_output_path=greyscale_saliency_path
            )
            
            # Check if user stopped during stage 3
            if shared_state.get('status') == 'stopped':
                print("[Pipeline Thread] User stopped during Stage 3.")
                return

            if not success:
                 if not shared_state.get('error'):
                    shared_state['error'] = "Stage 3 failed: Saliency map generation was unsuccessful."
                 shared_state['status'] = 'error'
                 return
        
        # Check for stop one last time before declaring completion
        if shared_state.get('status') == 'stopped':
             print("[Pipeline Thread] User stopped just before completion.")
             return
             
        shared_state['final_result_path'] = args.output
        shared_state['status'] = 'complete' # Signal success!

    except Exception as e:
        print(f"💥 A critical error occurred in the processing thread: {e}")
        shared_state['error'] = f"Critical error: {e}"
        shared_state['status'] = 'error'


# ==============================================================================
# SECTION 6: FINAL VISUALIZATION ASSEMBLY
# (This logic is from the 'else' block of the old main loop)
# ==============================================================================
def display_final_results_2x2(args, shared_state):
    """
    Loads all final assets and displays them in the 2x2 grid.
    """
    print("\n" + "="*60); print("🎉 ASSEMBLING FINAL RESULTS!"); print("="*60)
    
    WINDOW_NAME = "Saliency Analysis Results"
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    try:
        # --- 1. Load Top-Left Image (User Saliency) ---
        saliency_map = cv2.imread(shared_state['final_result_path'])
        if saliency_map is None: 
            print(f"❌ Error: Could not load final result from {shared_state['final_result_path']}.")
            raise IOError(f"Could not load {shared_state['final_result_path']}")
        cv2.putText(saliency_map, "What you focused at", (15, 35), FONT, 0.8, (255, 255, 255), 2)
        h_main, w_main = saliency_map.shape[:2] # Get main dimensions

        # --- 2. Load Top-Right Image (Model Prediction) ---
        model_prediction = cv2.imread("images/model_prediction.jpg")
        if model_prediction is not None:
            h_model, w_model = model_prediction.shape[:2]
            # Resize to match height of user map, preserving aspect ratio
            w_new_model = int(w_model * h_main / h_model) 
            model_prediction_resized = cv2.resize(model_prediction, (w_new_model, h_main))
            cv2.putText(model_prediction_resized, "Model prediction", (15, 35), FONT, 0.8, (255, 255, 255), 2)
        else:
            # Create a placeholder if model prediction is missing
            print("⚠️ Warning: 'model_prediction.jpg' not found.")
            w_new_model = int(w_main * 0.75) # Arbitrary width
            model_prediction_resized = np.zeros((h_main, w_new_model, 3), dtype=np.uint8)
            cv2.putText(model_prediction_resized, "Model prediction not found", (15, 35), FONT, 0.8, (255, 255, 255), 2)

        # --- 3. Create Bottom-Left Image (Gaze Scanpath) ---
        scanpath_image = cv2.imread(args.ref_image) 
        fixations_path = shared_state.get('fixations_mapped_path') 
        scanpath_resized_bl = None 
        
        if fixations_path and os.path.exists(fixations_path) and scanpath_image is not None:
            print("✅ Generating scanpath visualization...")
            try:
                df = pd.read_csv(fixations_path, sep='\t')
                fixations = df.groupby('fixation_id')[['ref_fixX', 'ref_fixY']].mean().dropna().sort_index()
                h_ref, w_ref = scanpath_image.shape[:2]
                points = fixations[
                    (fixations['ref_fixX'] >= 0) & (fixations['ref_fixX'] < w_ref) &
                    (fixations['ref_fixY'] >= 0) & (fixations['ref_fixY'] < h_ref)
                ].values.astype(int)

                if len(points) > 0:
                    LINE_COLOR = (0, 255, 255); START_COLOR = (0, 255, 0); END_COLOR = (0, 0, 255); MID_COLOR = (255, 0, 0)
                    POINT_RADIUS_LARGE = 12; POINT_RADIUS_SMALL = 8; TEXT_OFFSET_X = 15; TEXT_OFFSET_Y = 5
                    DESIRED_TIP_PIXELS = 16
                    
                    for i in range(len(points) - 1):
                        pt1 = points[i]; pt2 = points[i+1]
                        distance = np.linalg.norm(pt2 - pt1)
                        if distance == 0: continue
                        dynamic_tip_length = DESIRED_TIP_PIXELS / distance
                        cv2.arrowedLine(scanpath_image, tuple(pt1), tuple(pt2), LINE_COLOR, 2, tipLength=dynamic_tip_length)

                    if len(points) > 2:
                        for i in range(1, len(points) - 1):
                            cv2.circle(scanpath_image, tuple(points[i]), POINT_RADIUS_SMALL, MID_COLOR, -1) 

                    start_pt = tuple(points[0])
                    cv2.circle(scanpath_image, start_pt, POINT_RADIUS_LARGE, START_COLOR, 3) 
                    cv2.putText(scanpath_image, "Start", (start_pt[0] + TEXT_OFFSET_X, start_pt[1] + TEXT_OFFSET_Y), FONT, 0.7, START_COLOR, 2)

                    if len(points) > 1:
                        end_pt = tuple(points[-1])
                        cv2.circle(scanpath_image, end_pt, POINT_RADIUS_LARGE, END_COLOR, 3)
                        cv2.putText(scanpath_image, "End", (end_pt[0] + TEXT_OFFSET_X, end_pt[1] + TEXT_OFFSET_Y), FONT, 0.7, END_COLOR, 2)
                    
                    cv2.putText(scanpath_image, "Gaze Scanpath", (15, 35), FONT, 0.8, (255, 255, 255), 2)
                    cv2.putText(scanpath_image, f"{len(points)} fixations", (15, 70), FONT, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(scanpath_image, "Gaze Scanpath (No valid fixations)", (15, 35), FONT, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"⚠️ Error generating scanpath: {e}")
                scanpath_image = cv2.imread(args.ref_image) # Reset
                cv2.putText(scanpath_image, "Error generating scanpath", (15, 35), FONT, 0.8, (0, 0, 255), 2)
            
            scanpath_resized_bl = cv2.resize(scanpath_image, (w_main, h_main)) # Resize to top-left
        else:
            print("⚠️ Warning: Could not generate scanpath (missing data).")
            scanpath_resized_bl = np.zeros((h_main, w_main, 3), dtype=np.uint8)
            msg = "Scanpath data not found"
            if not fixations_path: msg = "Scanpath path missing"
            elif scanpath_image is None: msg = "Reference image missing"
            cv2.putText(scanpath_resized_bl, msg, (15, 35), FONT, 0.8, (255, 255, 255), 2)

        # --- 4. Create Bottom-Right Image (Diverging Difference Map) ---
        saliency_map_grey = cv2.imread("images/saliency_map_greyscale.jpg", cv2.IMREAD_GRAYSCALE)
        model_prediction_grey = cv2.imread("images/model_prediction_greyscale.jpg", cv2.IMREAD_GRAYSCALE)
        ref_image_for_diff = cv2.imread(args.ref_image)
        
        diff_map_resized = None # Initialize
        if saliency_map_grey is not None and model_prediction_grey is not None and ref_image_for_diff is not None:
            print("✅ Generating diverging difference map...")
            h_ref, w_ref = ref_image_for_diff.shape[:2]
            
            saliency_map_grey_r = cv2.resize(saliency_map_grey, (w_ref, h_ref))
            model_prediction_grey_r = cv2.resize(model_prediction_grey, (w_ref, h_ref))
            
            saliency_float = saliency_map_grey_r.astype(np.float32)
            model_float = model_prediction_grey_r.astype(np.float32)
            signed_diff = saliency_float - model_float
            diff_map_norm = ((signed_diff + 255.0) / 510.0) * 255.0
            diff_map_norm_u8 = diff_map_norm.astype(np.uint8)
            
            diff_map_color = cv2.applyColorMap(diff_map_norm_u8, cv2.COLORMAP_JET)
            diff_map_image = cv2.addWeighted(ref_image_for_diff, 0.5, diff_map_color, 0.5, 0)
            
            cv2.putText(diff_map_image, "Difference Map", (15, 35), FONT, 0.8, (255, 255, 255), 2)
            cv2.putText(diff_map_image, "Red = You, Blue = Model", (15, 70), FONT, 0.7, (255, 255, 255), 2)

            diff_map_resized = cv2.resize(diff_map_image, (w_new_model, h_main)) # Resize to top-right
        else:
            print("⚠️ Warning: Could not generate difference map.")
            diff_map_resized = np.zeros((h_main, w_new_model, 3), dtype=np.uint8)
            msg = "Difference map failed"
            if saliency_map_grey is None: msg = "Missing 'saliency_map_greyscale.jpg'"
            elif model_prediction_grey is None: msg = "Missing 'model_prediction_greyscale.jpg'"
            cv2.putText(diff_map_resized, msg, (15, 35), FONT, 0.8, (255, 255, 255), 2)
        
        # --- 5. Assemble Final 2x2 Grid ---
        top_row = np.hstack((saliency_map, model_prediction_resized))
        bottom_row = np.hstack((scanpath_resized_bl, diff_map_resized)) 
        full_composite_image = np.vstack((top_row, bottom_row))

        # --- 6. Resize the final composite image to fit the screen ---
        MAX_DISPLAY_HEIGHT = 900 # Max height in pixels
        h_full, w_full = full_composite_image.shape[:2]

        if h_full > MAX_DISPLAY_HEIGHT:
            print(f"ℹ️  Resizing final image from {w_full}x{h_full} to fit screen...")
            scale = MAX_DISPLAY_HEIGHT / h_full
            w_scaled = int(w_full * scale)
            final_display_image = cv2.resize(full_composite_image, (w_scaled, MAX_DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            final_display_image = full_composite_image
        
        # --- 7. Display Final Image ---
        cv2.imshow(WINDOW_NAME, final_display_image)
        print("✅ Final results are being displayed. Press any key to exit.")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"❌ Could not display final results: {e}")
        # Use placeholder method from waiting room class to show error
        error_img = InteractiveWaitingRoom("", "", "")._create_placeholder(f"Error: {e}", (400, 800))
        cv2.imshow("Error", error_img)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


# ==============================================================================
# SECTION 7: MAIN EXECUTION
# (This now uses the 'inter.py' orchestration style)
# ==============================================================================
if __name__ == "__main__":



    # 1. Argparse (from full.py)
    parser = argparse.ArgumentParser(description="Full pipeline from Pupil Labs export to saliency map.")
    parser.add_argument("-r", "--ref_image", default="images/reference_image.jpg", help="Path to the static reference image.")
    parser.add_argument("-o", "--output", default="images/saliency_map.png", help="Path to save the final saliency map.")
    parser.add_argument("-s", "--frame_step", type=int, default=1, help="Process every N-th frame for homography.")
    args = parser.parse_args()

    # 2. Asset Paths (for waiting room) 
    REFERENCE_IMAGE_PATH = args.ref_image
    MODEL_PREDICTION_PATH = "images/model_prediction.jpg"
    MODEL_PREDICTION_GREYSCALE_PATH = "images/model_prediction_greyscale.jpg"
    
    # 3. Shared State
    shared_state = {
        'status': 'running', # Main state for waiting room loop
        'error': None,
        'final_result_path': None,
        'fixations_mapped_path': None # For final 2x2 display
    }
    
    start_time = time.time()

    # 4. Start Pipeline Thread (using new full worker)
    pipeline_thread = threading.Thread(target=full_pipeline_worker, args=(args, shared_state), daemon=True)
    pipeline_thread.start()

    # 5. Start Waiting Room (this blocks the main thread)
    waiting_room = InteractiveWaitingRoom(
        REFERENCE_IMAGE_PATH,
        MODEL_PREDICTION_PATH,
        MODEL_PREDICTION_GREYSCALE_PATH
    )
    waiting_room.run(shared_state) # This blocks until status is 'complete', 'error', or 'stopped'

    # 6. Join Thread
    pipeline_thread.join() 

    # 7. Display Final Results
    if shared_state['status'] == 'complete':
        end_time = time.time(); total_time = end_time - start_time
        minutes, seconds = divmod(total_time, 60)
        print(f"\nTotal execution time: {int(minutes)} minutes and {seconds:.2f} seconds.")
        
        # Call the 2x2 grid display function
        display_final_results_2x2(args, shared_state) 
        
    elif shared_state['status'] == 'error':
        print("\n" + "="*60)
        print(f"❌ PIPELINE FAILED: {shared_state.get('error', 'Unknown error')}")
        print("="*60)
    else: # 'stopped'
        print("Execution was stopped by the user. No final results to display.")

    print("\nProgram finished.")
    sys.exit(0)