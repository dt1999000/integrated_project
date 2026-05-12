# 3D Object Detection Pipeline

Streamlit application for loading autonomous-driving datasets, running a 3D object detection pipeline, evaluating results, and exporting annotations and outputs.

## What This Project Does

The app is split into four Streamlit pages that share state through `st.session_state`:

1. `1_Dataset_Extraction` loads a sample or batch from KITTI, nuScenes, SUNRGBD, ROS bag, or sim data.
2. `2_Detection` runs the detection pipeline step by step or end to end.
3. `3_Evaluation` compares detections with ground truth and computes metrics.
4. `4_Export` writes results to JSON, KITTI-style tracklets, and CVAT-compatible exports.

## Recommended Workflow

Follow this order for the smoothest experience:

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Launch the Streamlit app from `app.py`.
5. Open `1_Dataset_Extraction` and set the dataset path plus output directory.
6. Load one sample or prepare a batch.
7. Run `2_Detection` on the loaded sample or batch.
8. Review metrics in `3_Evaluation`.
9. Export outputs from `4_Export`.

For batch workflows, the usual sequence is:

1. Prepare the batch on `1_Dataset_Extraction`.
2. Confirm the output directory and tracking settings.
3. Run batch detection on `2_Detection`.
4. Check the aggregate metrics in `3_Evaluation`.
5. Export the batch results from `4_Export`.
6. Upload the exported annotations to CVAT if manual correction is needed.

## Setup

### Clone The Repository

```bash
git clone https://github.com/dt1999000/integrated_project.git
cd integrated_project
```

### Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Notes:

- The project uses Streamlit, NumPy, OpenCV, Open3D, PyTorch, Ultralytics, and additional detection and segmentation packages.
- Some packages in `requirements.txt` are heavy and may take time to build or download.
- If you are on a machine without CUDA support, keep the default CPU-compatible settings where possible.

## Run The App

```bash
streamlit run app.py
```

This opens the main dashboard with the four pages in the sidebar.

## Models Preparation
You need to create the models directory and add the pretrained weights for the detection and segmentation models you want to use. The app will look for these files when running the pipeline. The structure of the models directory should be as follows:

```models/
├── sam*.pth # SAM segmentation model weights
├── yoloworld or yoloe*.pt # YOLOWorld/e detection model weights
├── llm
    |-- *.pth # LLM-based model weights
├── some external models weights from hugging face, e.g. grounding dino/wedetect, etc.
```
get at least a sam segmentation model from e.g. https://docs.ultralytics.com/models/sam-2/#installation and a yoloworld model from https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes
For batch runs, make sure the full set of weights is already present before you start processing. The same models are reused for every sample in the queue.

## How To Use Each Page

### Dataset Extraction

Set these inputs first:

- `Dataset Path`: root directory of the dataset you want to use.
- `Output folder for saved samples`: where extracted images and LiDAR files should be written.

Then choose the dataset-specific flow:

- KITTI: load a sample by numeric index.
- nuScenes: load a sample by token or use batch filtering.
- SUNRGBD: load the dataset and select a sample.
- ROS bag: configure topics, filter frames, and optionally extract a batch.
- sim: filter and sample linked data, then load the selected item.

Short download links:

- KITTI: https://www.cvlibs.net/datasets/kitti/
- nuScenes: https://www.nuscenes.org/nuscenes
- SUNRGBD: https://rgbd.cs.princeton.edu/ (3D data + annotation prep guide: https://mmdetection3d.readthedocs.io/en/v0.18.1/datasets/sunrgbd_det.html)
- ROS bag examples and format: https://github.com/ros2/rosbag2
- sim: custom/internal dataset format (no public canonical download)

The page stores the active dataset path, dataset type, calibration, and ground-truth annotations in session state for the later pages.

### Detection

Use this page to process the loaded sample or batch:

- Run the full pipeline for the quickest end-to-end result.
- Run the individual steps if you want to inspect intermediate outputs.
- Adjust clustering, cuboid fitting, and segmentation parameters in the sidebar.
- Reuse the calibration and metadata loaded on the extraction page.

For batch processing, the same pipeline is applied to every queued sample. If tracking is enabled, the batch output keeps instance IDs so detections can be followed across frames.

Typical order inside the pipeline is:

1. Ground plane removal.
2. Sparse depth backprojection.
3. SAM segmentation.
4. Clustering.
5. Detection and pose estimation.

### Evaluation

Open this page after detection to:

- Compare predicted cuboids with ground truth.
- Inspect 2D and 3D visualizations.
- Review IoU-based matching statistics.
- Analyze batch metrics and per-class summaries.

If you loaded a batch, the page also provides aggregate statistics and mismatch exports.

### Export

Use this page to write outputs to disk:

- JSON for detected 3D cuboids.
- JSON for 2D image annotations and bounding boxes.
- KITTI-style tracklet XML.
- CVAT-compatible annotation export.

Exports are written under the output directory selected on the extraction page.

The 3D detection exports can be uploaded into CVAT for manual correction. This is the recommended way to clean up missed detections, adjust cuboids, or fix labels after the automatic pipeline finishes.

Typical CVAT round-trip:

1. Export the 3D detection JSON or CVAT-compatible annotation file from `4_Export`.
2. Import the file into CVAT.
3. Correct the cuboids and labels manually.
4. Re-export the corrected annotations for downstream use.

## Project Layout

```text
app.py                      # Streamlit entry point
pages/                      # Streamlit pages
components/core/            # Detection, tracking, segmentation, projection, clustering
components/dataset_loaders/ # Dataset adapters and ROS bag extraction helpers
components/utils/           # Export and visualization helpers
models/                     # Pretrained model weights and configs
dataset/                    # Example/working data inputs
output/                     # Generated results and exports
```

## Practical Tips

- Start with a small sample before running large batches.
- Set the output directory before processing so results can be saved automatically.
- If calibration looks wrong, go back to `1_Dataset_Extraction` and reload the sample.
- For ROS bags and batch workflows, prefer the batch path first and only export after the batch finishes.
- If you plan to correct detections manually, export the batch results in a CVAT-friendly format before opening them in CVAT.

## Troubleshooting

- If the app cannot find a dataset, verify the path exists and points to the dataset root.
- If a model fails to load, confirm the matching weight file is present under `models/`.
- If you see missing dependency errors, reinstall the environment with `pip install -r requirements.txt` inside a clean virtual environment.
- If Streamlit keeps stale state, reload the app after changing dataset paths or calibration settings.

## License

Add the project license here if one applies.