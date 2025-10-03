#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full pipeline to automate Pupil Labs data processing and generate a saliency map.

This script orchestrates three main stages:
1.  Automates the Pupil Player export process for the latest recording.
2.  Maps the exported gaze and fixation data from the world video onto a static reference image.
3.  Generates and displays a saliency map (heatmap) based on the mapped gaze data.

Prerequisites:
- Python libraries: opencv-python, numpy, pandas, scipy
- Pupil Labs software installed and configured.
- A 'config.ini' file in the same directory as this script.
- 'xdotool' command-line utility installed (`sudo apt-get install xdotool` on Debian/Ubuntu).

Usage:
    python full_pipeline.py /path/to/your/reference_image.jpg -o /path/to/save/saliency_map.png
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

# --- Scientific and Image Processing Imports ---
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import gaussian_kde

# ==============================================================================
# SECTION 1: PUPIL LABS EXPORT AUTOMATION (from pipeline_pupil.py)
# ==============================================================================

def run_pupil_pipeline():
    """
    Finds the latest recording, launches Pupil Player, triggers an export,
    and returns the path to the exported data directory.
    """
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
        print(f"❌ Error: 'config.ini' is missing a section or key: {e}")
        return None

    # --- 2. Determine Today's Recording Directory ---
    today_str = time.strftime('%Y_%m_%d')
    daily_folder_path = os.path.join(base_recordings_dir, today_str)
    
    print(f"2. ▶️  Checking for recordings in: {daily_folder_path}")
    if not os.path.isdir(daily_folder_path):
        print(f"❌ Error: Today's recording directory not found.")
        return None

    # --- 3. Identify the Latest Recording ---
    try:
        numeric_dirs = [d for d in os.listdir(daily_folder_path) if os.path.isdir(os.path.join(daily_folder_path, d)) and d.isdigit()]
        if not numeric_dirs: raise FileNotFoundError
        latest_recording_subdir = max(numeric_dirs, key=int)
        latest_recording_dir = os.path.join(daily_folder_path, latest_recording_subdir)
    except (OSError, FileNotFoundError):
        print(f"❌ Error: No valid recordings found in '{daily_folder_path}'.")
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
    print("5. ⏳ Launching Pupil Player... (this may take a moment)")
    pupil_process = subprocess.Popen(
        pupil_player_command, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, preexec_fn=os.setsid
    )
    
    # --- 6. Wait for Initial Processing to Complete ---
    print("6. 👀 Waiting for fixation detection signal (timeout: 5s of inactivity)...")

    processing_complete = False
    # We use this flag to know if we saw at least one "ready" signal before timing out.
    # This makes our "success" message more accurate.
    any_ready_signal_seen = False

    while True:
        # select.select() waits until the stdout stream has data to be read.
        # It will wait for a maximum of 5.0 seconds.
        # - If data arrives, it returns a list like: [pupil_process.stdout]
        # - If 5.0s pass with no data, it returns an empty list: []
        ready_to_read, _, _ = select.select([pupil_process.stdout], [], [], 5.0)

        # CASE 1: TIMEOUT - No new output for 5 seconds.
        if not ready_to_read:
            if any_ready_signal_seen:
                print("   -> ✅ Timeout reached after last output. Assuming player is ready.")
            else:
                print("   -> ⚠️  Timeout reached with no initial output. Assuming player is ready anyway.")
            processing_complete = True
            break

        # CASE 2: NEW OUTPUT - A new line is available to be read.
        line = pupil_process.stdout.readline()
        
        # If the line is empty, it means the subprocess has closed.
        if not line:
            print("   -> Pupil Player process stream ended unexpectedly.")
            processing_complete = any_ready_signal_seen
            break

        stripped_line = line.strip()
        if stripped_line:
            print(f"   | {stripped_line}")

        # Check for either of the "ready" signals.
        if "World Video Exporter has been launched." in line:
            any_ready_signal_seen = True

        # The main condition: if we see the fixation data, we can stop waiting and proceed immediately.
        if "Starting fixation detection using 3d gaze data" in line:
            print("   -> ✅ Found fixation detection signal. Continuing immediately.")
            time.sleep(1)
            any_ready_signal_seen = True
            processing_complete = True
            break

    # --- Final check to ensure we can proceed ---
    if not processing_complete:
        print("\n❌ Error: Player did not initialize correctly or closed before it was ready.")
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None

    print("6. ✅ Initial processing complete. Player is ready for export command.")



    # --- 7. Trigger the Export ---

    print("7. ⏳ Waiting for Pupil Player window to appear...")
    # Define the unique, static part of the window name to search for
    window_name_pattern = "Pupil Player:"
    
    # Robustly wait for the window to exist
    window_found = False
    max_wait_seconds = 30
    for i in range(max_wait_seconds * 2): # Check every 0.5 seconds
        check_window_command = f'xdotool search --onlyvisible --name "{window_name_pattern}"'
        result = subprocess.run(check_window_command, shell=True, capture_output=True, executable='/bin/bash')
        
        if result.returncode == 0:
            print(f"7.1 ✅ Found Pupil Player window matching '{window_name_pattern}'.")
            window_found = True
            break
        
        time.sleep(0.5)

    if not window_found:
        print(f"   ❌ Error: Pupil Player window did not appear within {max_wait_seconds} seconds.")
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None

    # Now that the window exists, send the keypress
    print("7.2 ⌨️ Simulating 'e' key press to trigger export...")
    try:
        time.sleep(1) 
        keypress_command = f'xdotool search --onlyvisible --name "{window_name_pattern}" windowactivate --sync key e'
        subprocess.run(keypress_command, shell=True, check=True, capture_output=True, text=True, executable='/bin/bash')
        print("   ✅ Keypress 'e' sent successfully.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"   ❌ Error: Failed to send keypress. Is 'xdotool' installed? Reason: {e}")
        os.killpg(os.getpgid(pupil_process.pid), signal.SIGKILL)
        return None


    # --- 8. Wait for Export to Finish ---
    print("8. 👀 Monitoring for 'Export done' confirmation...")
    export_confirmed = False
    for line in pupil_process.stdout:
        # Uncomment the line below for verbose output
        # print(line.strip())
        if "Export done: Exported" in line:
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
        else:
            print("   ⚠️ Could not find a new export folder on the filesystem.")
    else:
        print("\n   ❌ Error: 'Export done' message was not detected.")

    # --- 10. Close Pupil Player ---
    print("10. 🚮 Closing Pupil Player application...")
    try:
        pgid = os.getpgid(pupil_process.pid)
        
        # First, politely ask the entire process group to terminate
        print("   Sending SIGTERM for graceful shutdown...")
        os.killpg(pgid, signal.SIGTERM)
        
        # Wait a few seconds to allow for a graceful shutdown
        wait_seconds = 4
        time.sleep(wait_seconds)
        
        # Forcefully kill any remaining processes in the group as a fallback
        print("   Sending SIGKILL to ensure all processes are terminated.")
        os.killpg(pgid, signal.SIGKILL)
        
        print("   ✅ Pupil Player closed successfully.")
    except ProcessLookupError:
        print("   ✅ Pupil Player was already closed.")
    except Exception as e:
        print(f"   ⚠️ An error occurred while closing Pupil Player: {e}")

    return os.path.abspath(final_export_path)


# ==============================================================================
# SECTION 2: GAZE MAPPING (from mapGaze.py)
# ==============================================================================

# --- Homography and Feature Matching Functions ---

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

def preprocess_frame(frame):
    frame = adjust_gamma(frame, gamma=1.5)
    frame = local_brightness_norm(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    return gray, None # Simplified from original

def find_matches(kp1, des1, kp2, des2, dist_ratio=0.7, min_good_matches=12):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if des1 is None or des2 is None: return None, None
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < dist_ratio * n.distance]
    if len(good) >= min_good_matches:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2
    return None, None

def compute_homography(pts_ref, pts_frame, ransac_thresh=8.0):
    H, mask = cv2.findHomography(
        pts_ref.reshape(-1,1,2), pts_frame.reshape(-1,1,2),
        cv2.RANSAC, ransac_thresh
    )
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
    gray, _ = preprocess_frame(frame)
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

# --- Main Gaze Mapping Orchestrator ---

def run_map_gaze(export_dir, ref_image_path):
    """
    Processes video from an export folder to map gaze onto a reference image.
    """
    print("\n" + "="*60)
    print("🖼️  STAGE 2: MAPPING GAZE TO REFERENCE IMAGE")
    print("="*60)

    # --- Define Paths ---
    gaze_csv = os.path.join(export_dir, 'gaze_positions.csv')
    fixations_csv = os.path.join(export_dir, 'fixations.csv')
    video_path = os.path.join(export_dir, 'world.mp4')
    out_dir = os.path.join(export_dir, 'mapGaze_output')
    
    # --- Validate Inputs ---
    for path in [gaze_csv, fixations_csv, video_path, ref_image_path]:
        if not os.path.exists(path):
            print(f"❌ Error: Required file not found at {path}")
            return None
    
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(ref_image_path, out_dir)
    print(f"✅ Output will be saved to: {out_dir}")

    # --- Load Data ---
    gaze_df = pd.read_csv(gaze_csv)
    fixations_df = pd.read_csv(fixations_csv)

    # --- Normalize Frame Indices ---
    min_frame_index = gaze_df['world_index'].min()
    gaze_df['world_index'] -= min_frame_index
    fixations_df['start_frame_index'] -= min_frame_index
    fixations_df['end_frame_index'] -= min_frame_index

    # --- Setup Video and Feature Detector ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_img = cv2.imread(ref_image_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    detector = cv2.AKAZE_create()
    ref_kp, ref_des = detector.detectAndCompute(ref_gray, None)

    idx = 0
    gaze_mapped_records = []
    fixations_mapped_records = []
    
    print(f"⏳ Processing {total_frames} video frames...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        Hs = process_single_frame(frame, ref_kp, ref_des, detector, ref_gray.shape)
        
        if Hs:
            # Process gaze data for this frame
            frame_gaze = gaze_df[gaze_df['world_index'] == idx]
            for _, row in frame_gaze.iterrows():
                wx = row['norm_pos_x'] * frame.shape[1]
                wy = (1 - row['norm_pos_y']) * frame.shape[0]
                rx, ry = map_coords((wx, wy), Hs['H_frame2ref'])
                gaze_mapped_records.append({
                    'worldFrame': idx, 'gaze_ts': row['gaze_timestamp'], 'confidence': row['confidence'],
                    'world_gazeX': wx, 'world_gazeY': wy, 'ref_gazeX': rx, 'ref_gazeY': ry
                })
            
            # Process fixations that span this frame
            this_frame_fixations = fixations_df[(fixations_df['start_frame_index'] <= idx) & (fixations_df['end_frame_index'] >= idx)]
            for _, fix_row in this_frame_fixations.iterrows():
                fix_wx = fix_row['norm_pos_x'] * frame.shape[1]
                fix_wy = (1 - fix_row['norm_pos_y']) * frame.shape[0]
                fix_rx, fix_ry = map_coords((fix_wx, fix_wy), Hs['H_frame2ref'])
                fixations_mapped_records.append({
                    'fixation_id': fix_row['id'], 'start_ts': fix_row['start_timestamp'], 'duration': fix_row['duration'],
                    'confidence': fix_row['confidence'], 'worldFrame': idx,
                    'world_fixX': fix_wx, 'world_fixY': fix_wy, 'ref_fixX': fix_rx, 'ref_fixY': fix_ry
                })

        if idx % 100 == 0:
            print(f"   ... processed frame {idx}/{total_frames}")
        idx += 1

    cap.release()
    print("✅ Video processing complete.")

    # --- Save Mapped Data ---
    gaze_mapped_df = pd.DataFrame(gaze_mapped_records)
    fixations_mapped_df = pd.DataFrame(fixations_mapped_records)
    
    gaze_output_path = os.path.join(out_dir, 'gazeData_mapped.tsv')
    fixations_output_path = os.path.join(out_dir, 'fixations_mapped.tsv')
    
    gaze_mapped_df.to_csv(gaze_output_path, sep='\t', index=False)
    fixations_mapped_df.to_csv(fixations_output_path, sep='\t', index=False)
    
    print(f"✅ Mapped gaze data saved to {gaze_output_path}")
    print(f"✅ Mapped fixations data saved to {fixations_output_path}")
    
    return out_dir


# ==============================================================================
# SECTION 3: SALIENCY MAP GENERATION
# ==============================================================================

def create_gaussian_heatmap(x_coords, y_coords, width, height, sigma=50):
    """Create a heat map using Gaussian blurring around each gaze point."""
    heat_map = np.zeros((height, width), dtype=np.float64)
    for x, y in zip(x_coords, y_coords):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < width and 0 <= y < height:
            heat_map[y, x] += 1
    return ndimage.gaussian_filter(heat_map, sigma=sigma)

def create_kde_heatmap(x_coords, y_coords, width, height):
    """Create a heat map using Kernel Density Estimation."""
    x_grid = np.linspace(0, width - 1, width // 4)
    y_grid = np.linspace(0, height - 1, height // 4)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_coords, y_coords])
    kernel = gaussian_kde(values)
    density = kernel(positions).reshape(xx.shape)
    return cv2.resize(density, (width, height))

def create_saliency_map(jpg_file, gaze_data_file, output_path, method="gaussian", sigma=50, alpha=0.6):
    """Create a saliency map from gaze data and overlay it on an image."""
    print("\n" + "="*60)
    print("🔥 STAGE 3: GENERATING SALIENCY MAP")
    print("="*60)

    try:
        img = cv2.imread(jpg_file)
        if img is None:
            print(f"❌ Error: Could not load background image from {jpg_file}")
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
            return False
        
        x_coords = valid_gaze['ref_gazeX'].values
        y_coords = valid_gaze['ref_gazeY'].values
        
        print(f"⏳ Creating heatmap from {len(x_coords)} valid gaze points...")
        
        if method == "gaussian":
            heat_map = create_gaussian_heatmap(x_coords, y_coords, w, h, sigma)
        elif method == "kde":
            heat_map = create_kde_heatmap(x_coords, y_coords, w, h)
        else:
            raise ValueError("Method must be 'gaussian' or 'kde'")
        
        if heat_map.max() == 0:
            print("⚠️ Warning: Heatmap is empty after processing points.")
            return False

        heat_map = (heat_map / heat_map.max() * 255).astype(np.uint8)
        heat_map_colored = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 1 - alpha, heat_map_colored, alpha, 0)
        
        cv2.imwrite(output_path, result)
        print(f"✅ Saliency map saved to {output_path}")
        return True

    except Exception as e:
        print(f"❌ An error occurred during saliency map generation: {e}")
        return False


# ==============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# ==============================================================================

if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Full pipeline from Pupil Labs export to saliency map.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-r", "--ref_image",
        help="Path to the static reference image for gaze mapping.",
        default="reference_image.jpg"  # The script will use this file if none is provided
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the final saliency map PNG file.",
        default="saliency_map.png"
    )
    args = parser.parse_args()

    # --- STEP 1: Run Pupil Preprocessing ---
    export_directory = run_pupil_pipeline()
    if not export_directory:
        print("\n❌ Pipeline halted: Failed to get Pupil Player export directory.")
        sys.exit(1)
    print(f"\n✅ STAGE 1 COMPLETE. Exported data is at: {export_directory}")

    # --- STEP 2: Run Gaze Mapping ---
    mapgaze_output_dir = run_map_gaze(export_directory, args.ref_image)
    if not mapgaze_output_dir:
        print("\n❌ Pipeline halted: Gaze mapping stage failed.")
        sys.exit(1)
    print(f"\n✅ STAGE 2 COMPLETE. Mapped data is at: {mapgaze_output_dir}")

    # --- STEP 3: Generate Saliency Map ---
    gaze_mapped_file = os.path.join(mapgaze_output_dir, 'gazeData_mapped.tsv')
    success = create_saliency_map(
        jpg_file=args.ref_image,
        gaze_data_file=gaze_mapped_file,
        output_path=args.output
    )
    if not success:
        print("\n❌ Pipeline halted: Saliency map generation failed.")
        sys.exit(1)
    print(f"\n✅ STAGE 3 COMPLETE.")

    # --- STEP 4: Display the Result ---
    print("\n" + "="*60)
    print("🎉 PIPELINE FINISHED SUCCESSFULLY!")
    print("="*60)
    end_time = time.time()
    total_time = end_time - start_time
    # Format the time into minutes and seconds for readability
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print("-" * 50)
    print(f"--------> Script finished in {minutes} minutes and {seconds:.2f} seconds.")
    
    try:
        # Load the primary image (the generated saliency map)
        saliency_map = cv2.imread(args.output)
        if saliency_map is None:
            print(f"Could not read the generated saliency map at {args.output} to display it.")
            sys.exit(1)

        # Attempt to load the comparison image
        model_img_path = "model_prediction.jpg"
        model_prediction = cv2.imread(model_img_path)
        
        # --- Define properties for the text labels ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # White color
        thickness = 2
        position = (15, 35)  # Position from the top-left corner (x, y)
        
        display_image = saliency_map
        window_name = "Saliency Map (Press 'q' or ESC to exit)"

        if model_prediction is not None:
            print(f"✅ Found '{model_img_path}'. Displaying side-by-side comparison.")
            # --- Resize logic (remains the same) ---
            h1, w1 = saliency_map.shape[:2]
            h2, w2 = model_prediction.shape[:2]
            if h1 != h2:
                new_w2 = int(w2 * (h1 / h2))
                model_prediction_resized = cv2.resize(model_prediction, (new_w2, h1), interpolation=cv2.INTER_AREA)
            else:
                model_prediction_resized = model_prediction

            # --- Add text labels to each image BEFORE combining ---
            cv2.putText(saliency_map, "What you focused at", position, font, font_scale, font_color, thickness)
            cv2.putText(model_prediction_resized, "Model prediction", position, font, font_scale, font_color, thickness)

            # Combine the labeled images horizontally
            display_image = np.hstack((saliency_map, model_prediction_resized))
            window_name = "Comparison (Press 'q' or ESC)"
        else:
            print(f"⚠️  Note: '{model_img_path}' not found. Showing only the generated saliency map.")
            # --- Add text label to the single image ---
            cv2.putText(saliency_map, "What you focused at", position, font, font_scale, font_color, thickness)
            display_image = saliency_map

        # Display the final image (either single or combined)
        cv2.imshow(window_name, display_image)
        
        # Wait until the user presses 'q' or the Escape key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        print("Window closed. Exiting.")

    except Exception as e:
        print(f"An error occurred while trying to display the image: {e}")



