# Pupil Labs Data Processing Pipeline

This repository contains a Python script (`pipeline_pupil.py`) that automates the processing of eye-tracking data collected using **Pupil Labs Capture** software with **Pupil Core** eye-tracking glasses.

The pipeline runs all processing (export, mapping, saliency generation) in a **background thread**. While the pipeline runs, it displays an **interactive "waiting room"** window, allowing the user to explore saliency concepts.

Once processing is complete, the script presents a **final 2x2 analysis grid** comparing the generated human saliency map, a neural network (NN) model's prediction, the gaze scanpath, and a diverging difference map (Was changed for deformation map based on saliency).

## Features

-   **Automated Pupil Player Export**: Automatically finds the latest recording, launches Pupil Player, and exports gaze and fixation data.
-   **Gaze Mapping**: Maps gaze and fixation data from the world video to a user-provided reference image using homography.
-   **Saliency Map Generation**: Creates a heatmap (saliency map) from the mapped gaze data.
-   **Interactive Waiting Room**: While the pipeline runs in the background, an OpenCV window provides multiple modes to explore saliency:
    -   **Tunnel Vision**: Simulates foveated vision by blurring the periphery.
    -   **Saliency Overlay**: Blends the model's saliency prediction over the reference image with an adjustable slider.
    -   **Deform**: A demo mode to warp the image with mouse clicks.
    -   **A/B Test**: A game to guess hotspots on the reference image and get a score based on the model's prediction.
-   **Final 2x2 Visualization**: Displays a comprehensive four-quadrant analysis grid upon completion:
    -   **Top-Left**: Generated human saliency map.
    -   **Top-Right**: Model-predicted saliency map.
    -   **Bottom-Left**: Gaze scanpath overlay showing the sequence of fixations.
    -   **Bottom-Right**: Diverging difference map highlighting areas where human and model saliency differed.

## Prerequisites

To run the pipeline, ensure the following are installed and configured:

1.  **Pupil Labs Software**:
    -   Install **Pupil Player/Capture** from the [Pupil Labs GitHub repository](https://github.com/pupil-labs/pupil).
    -   Set up a Python virtual environment for Pupil Labs software as described in the repository.
    -   Update the `config.ini` file with the correct paths:
        -   `pupil_project_dir`: Path to the Pupil Labs project directory containing `recordings` and `pupil_src` subfolders.
        -   `venv_activate_path`: Path to the `activate` script in the Pupil Labs virtual environment.

2.  **System Dependencies**:
    -   Install `xdotool` for automating keypresses in Pupil Player:
        ```bash
        sudo apt-get install xdotool
        ```
        (Required for Debian/Ubuntu systems; adjust for other operating systems.)

3.  **Python Libraries**:
    -   Install the required Python libraries listed in `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
        The file specifies:
        -   `numpy==2.3.3`
        -   `pandas==2.3.3`
        -   `scipy==1.16.2`
        -   `opencv-python==4.9.0.80`

4.  **Input Files**:
    -   A reference image (e.g., `images/reference_image.jpg`) to map gaze data onto.
    -   A model-predicted saliency map (default: `images/model_prediction.jpg`) for comparison.
    -   A **greyscale** version of the model-predicted saliency map (default: `images/model_prediction_greyscale.jpg`) used for the difference map and waiting room.

5.  **Configuration File**:
    -   Ensure a `config.ini` file exists in the same directory as the script, with the following structure:
        ```ini
        [Paths]
        pupil_project_dir = /path/to/your/pupil/project
        venv_activate_path = /path/to/your/pupil_env/bin/activate
        ```
        Update the paths to match your system setup.

## Usage

1.  **Collect Data**:
    -   Use **Pupil Capture** with Pupil Core eye-tracking glasses to record eye-tracking data.
    -   Recordings are saved in the `recordings` folder within the Pupil Labs project directory, typically organized by date (e.g., `YYYY_MM_DD`).

2.  **Run the Script**:
    -   Execute the script with the following command:
        ```bash
        python pipeline_pupil.py -r images/reference_image.jpg -o saliency_map.png -s 1
        ```
        -   `-r` or `--ref_image`: Path to the reference image (default: `images/reference_image.jpg`).
        -   `-o` or `--output`: Path to save the generated saliency map (default: `saliency_map.png`).
        -   `-s` or `--frame_step`: Process every N-th frame for homography (default: 1).

3.  **Pipeline Stages**:
    -   **Stage 1: Background Processing**: The script immediately starts a background thread that:
        1.  Automates Pupil Player to export data from the latest recording.
        2.  Maps gaze and fixation data from the world video to the reference image.
        3.  Generates the human saliency map and a greyscale version, saving them to disk.
    -   **Stage 2: Interactive Waiting Room**: While Stage 1 runs, an interactive OpenCV window is displayed.
        -   The window shows the current status of the background pipeline.
        -   Press `'1'` (Tunnel), `'2'` (Overlay), `'3'` (Deform), `'4'` (A/B Test) to switch modes.
        -   Press `'r'` to reset the current mode's state (e.g., reset A/B test clicks).
        -   Press `'q'` to stop the pipeline and exit the program.
    -   **Stage 3: Final 2x2 Analysis**: Once the pipeline thread is complete, the waiting room closes and a final window appears, displaying the 2x2 analysis grid.
        -   Press any key to close the final window and end the program.

4.  **Output**:
    -   The pipeline creates a `mapGaze_output` directory in the export folder containing:
        -   `gazeData_mapped.tsv`: Mapped gaze data.
        -   `fixations_mapped.tsv`: Mapped fixation data.
        -   A copy of the reference image.
    -   The final human saliency map is saved to the specified output path (e.g., `saliency_map.png`).
    -   A greyscale version of the human saliency map is saved (e.g., `images/saliency_map_greyscale.jpg`).
    -   The script displays the final 2x2 analysis grid.

## Example

```bash
python pipeline_pupil.py -r images/reference_image.jpg -o saliency_map.png
```
or just
```bash
python pipeline_pupil.py
```

This command:
- Displays the Interactive Waiting Room.
- In the background, processes the latest recording in the Pupil Labs recordings directory.
- Maps gaze data to images/reference_image.jpg.
- Generates and saves a saliency map to saliency_map.png (and images/saliency_map_greyscale.jpg).
- Once complete, closes the waiting room and displays the Final 2x2 Analysis Grid (using images/model_prediction.jpg and other generated files).

## Notes

- Ensure the Pupil Labs virtual environment is correctly set up and the `venv_activate_path` points to the `activate` script.
- The script assumes recordings are stored in a date-based folder structure (e.g., `recordings/YYYY_MM_DD/NNN`).
- If images/model_prediction.jpg or images/model_prediction_greyscale.jpg are not found, placeholders will be shown in the waiting room and the final 2x2 grid.
- The saliency map is generated using a Gaussian method by default (configurable via code modification to use KDE).
- The script includes error handling and verbose output for debugging.

## Troubleshooting

- **Missing `config.ini`**: Ensure the file exists and contains valid paths.
- **Pupil Player Fails to Launch**: Verify the `pupil_project_dir` and `venv_activate_path` in `config.ini`.
- **No Valid Recordings**: Check that recordings exist in the `recordings/YYYY_MM_DD` directory.
- **xdotool Errors**: Install `xdotool` and ensure the Pupil Player window is visible.
- **Empty Saliency Map**: Ensure the reference image matches the scene in the world video, and gaze data is valid.
