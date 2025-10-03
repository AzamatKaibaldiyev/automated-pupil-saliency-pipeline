# Pupil Labs Data Processing Pipeline

This repository contains a Python script (`pipeline_pupil.py`) that automates the processing of eye-tracking data collected using **Pupil Labs Capture** software with **Pupil Core** eye-tracking glasses. The pipeline processes raw gaze data, exports it using Pupil Player, maps gaze points to a static reference image, generates a saliency map, and displays it alongside an optional neural network (NN) model-predicted saliency map for comparison.

## Features

- **Automated Pupil Player Export**: Automatically finds the latest recording, launches Pupil Player, and exports gaze and fixation data.
- **Gaze Mapping**: Maps gaze and fixation data from the world video to a user-provided reference image using homography.
- **Saliency Map Generation**: Creates a heatmap (saliency map) from the mapped gaze data using either Gaussian or KDE methods.
- **Visualization**: Displays the generated saliency map, optionally side-by-side with a model-predicted saliency map (`model_prediction.jpg`).

## Prerequisites

To run the pipeline, ensure the following are installed and configured:

1. **Pupil Labs Software**:
   - Install **Pupil Player/Capture** from the [Pupil Labs GitHub repository](https://github.com/pupil-labs/pupil).
   - Set up a Python virtual environment for Pupil Labs software as described in the repository.
   - Update the `config.ini` file with the correct paths:
     - `pupil_project_dir`: Path to the Pupil Labs project directory containing `recordings` and `pupil_src` subfolders.
     - `venv_activate_path`: Path to the `activate` script in the Pupil Labs virtual environment.

2. **System Dependencies**:
   - Install `xdotool` for automating keypresses in Pupil Player:
     ```bash
     sudo apt-get install xdotool
     ```
     (Required for Debian/Ubuntu systems; adjust for other operating systems.)

3. **Python Libraries**:
   - Install the required Python libraries listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
     The file specifies:
     - `numpy==2.3.3`
     - `pandas==2.3.3`
     - `scipy==1.16.2`
     - `opencv-python==4.9.0.80`

4. **Input Files**:
   - A reference image (e.g., `reference_image.jpg`) to map gaze data onto.
   - Optionally, a model-predicted saliency map (`model_prediction.jpg`) for side-by-side comparison.

5. **Configuration File**:
   - Ensure a `config.ini` file exists in the same directory as the script, with the following structure:
     ```ini
     [Paths]
     pupil_project_dir = /path/to/your/pupil/project
     venv_activate_path = /path/to/your/pupil_env/bin/activate
     ```
     Update the paths to match your system setup.

## Usage

1. **Collect Data**:
   - Use **Pupil Capture** with Pupil Core eye-tracking glasses to record eye-tracking data.
   - Recordings are saved in the `recordings` folder within the Pupil Labs project directory, typically organized by date (e.g., `YYYY_MM_DD`).

2. **Run the Script**:
   - Execute the script with the following command:
     ```bash
     python pipeline_pupil.py -r /path/to/reference_image.jpg -o /path/to/save/saliency_map.png
     ```
     - `-r` or `--ref_image`: Path to the reference image (default: `reference_image.jpg`).
     - `-o` or `--output`: Path to save the generated saliency map (default: `saliency_map.png`).

3. **Pipeline Stages**:
   - **Stage 1**: Automates Pupil Player to export gaze and fixation data from the latest recording.
   - **Stage 2**: Maps gaze and fixation data from the world video to the reference image.
   - **Stage 3**: Generates a saliency map using the mapped gaze data and saves it to the specified output path.
   - **Stage 4**: Displays the saliency map. If `model_prediction.jpg` exists, it is shown side-by-side with the generated saliency map.

4. **Output**:
   - The pipeline creates a `mapGaze_output` directory in the export folder containing:
     - `gazeData_mapped.tsv`: Mapped gaze data.
     - `fixations_mapped.tsv`: Mapped fixation data.
     - A copy of the reference image.
   - The final saliency map is saved to the specified output path (e.g., `saliency_map.png`).
   - The script displays the saliency map (and model prediction, if available) in a window. Press `q` or `ESC` to close the window.

## Example

```bash
python pipeline_pupil.py -r reference_image.jpg -o saliency_map.png
```
or just
```bash
python pipeline_pupil.py
```

This command:
- Processes the latest recording in the Pupil Labs `recordings` directory.
- Maps gaze data to `reference_image.jpg`.
- Generates and saves a saliency map to `saliency_map.png`.
- Displays the saliency map alongside `model_prediction.jpg` (if it exists).

## Notes

- Ensure the Pupil Labs virtual environment is correctly set up and the `venv_activate_path` points to the `activate` script.
- The script assumes recordings are stored in a date-based folder structure (e.g., `recordings/YYYY_MM_DD/NNN`).
- If `model_prediction.jpg` is not found, only the generated saliency map is displayed.
- The saliency map is generated using a Gaussian method by default (configurable via code modification to use KDE).
- The script includes error handling and verbose output for debugging.

## Troubleshooting

- **Missing `config.ini`**: Ensure the file exists and contains valid paths.
- **Pupil Player Fails to Launch**: Verify the `pupil_project_dir` and `venv_activate_path` in `config.ini`.
- **No Valid Recordings**: Check that recordings exist in the `recordings/YYYY_MM_DD` directory.
- **xdotool Errors**: Install `xdotool` and ensure the Pupil Player window is visible.
- **Empty Saliency Map**: Ensure the reference image matches the scene in the world video, and gaze data is valid.
