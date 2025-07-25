# Dispatch Vision System

A computer vision-based inspection system designed to monitor and verify food dispatch workflows.  
This project applies object detection techniques to detect trays, dishes, and serving statuses (e.g., empty / not empty) to ensure quality and accuracy before meals are delivered.

---

## Features

- **Object Detection** of trays, dishes, and food items
- **Classification** of serving status (empty / not empty / kakigori)
- Visual output with annotated bounding boxes
- Real-time or batch-mode processing
- Modular and easy to integrate into production pipelines

---

## Project Structure
```
dispatch-vision-system/
│
├── feedback_data/               # User feedback for model correction
│   └── feedback.csv
│
├── modules/                     # Core processing and model modules
│   ├── classifier.py            # Classification logic (e.g., empty / not_empty / kakigori)
│   ├── frame_processor.py       # Frame-level pipeline control
│   ├── preprocess_image_cls.py  # Preprocessing for classification
│   └── preprocess_image_det.py  # Preprocessing for detection
│
├── results/                     # Detection and feedback visualization outputs
│
├── tray_dish_detection_classification.ipynb  # Jupyter demo notebook
├── main.py                      # Main entry point to run the system
├── Dockerfile                   # Docker container setup
├── docker-compose.yml           # Docker Compose configuration for running the system
├── requirements.txt             # Python dependencies
```

# Installation Guide and Usage Instructions
## 1. Install all required packages.
```bash
pip install -r requirements.txt
```
## 2. Training.
### The training configuration is located in the tray_dish_detection_classification.ipynb file. With two steps:
1. Train an object detection model to identify trays and dishes.
2. Train a classification model to categorize the detected objects into three classes: "not_empty", "empty", and "kakigori".
After training, three models are generated: best.pth (detection model), cls_dish_best.pth, and cls_tray_best.pth (classification models).

## 3. Evaluation on the validation dataset.
### 3.1. The valid detection result.
<img src="results/detection_result.png" alt="Detection Result" width="700"/>

### 3.2. The classification results.
| Object Type | Best Validation Accuracy |
|-------------|--------------------------|
| Tray        | 96.68%                   |
| Dish        | 96.46%                   |

## 4. Inference.
### 4.1. Load streamlit tool.
The project is developed using the Streamlit tool. To execute the project:
```bash
docker pull xuanhiepp/kitchen-inspection-full:latest
docker run -p 8501:8501 xuanhiepp/kitchen-inspection-full:latest
```
Since the port is mapped as 8501:8501 in docker-compose.yml, you can open your browser and access the application at:
http://localhost:8501 or http://127.0.0.1:8501
### 4.2. Launch the Streamlit app to use the following features.
**Feature 1:** The left column displays the prediction results in image format, with bounding boxes and corresponding object labels.

**Feature 2:** The right column presents detailed information, including bounding box coordinates, detection labels, and classification labels.

<img src="results/project_result.png" alt="Project Result"/><br>

**Feature 3:** The right section provides a feedback feature where users can correct detection and classification labels. The feedback is saved to feedback.csv in the feedback_data directory, recording timestamps, frame indices, object coordinates, predicted and corrected labels, and prediction confidence. To access the file directly inside the container:

**Step 1:** Get the running container ID:
```bash
docker ps
```
For example, the container ID may be:
```bash
CONTAINER ID   456yhf123mnk   kitchen-inspection-full:latest
```
**Step 2:** Enter the container:
```bash
docker exec -it 456yhf123mnk /bin/bash
```
**Step 3:** Navigate to the feedback directory and check the file:
```bash
cd /app/feedback_data/
ls
cat feedback.csv
```
Users can correct both detection and classification results through the following interface:

<img src="results/feedback_result.png" alt="Feedback Result" width="350"/>

All submitted feedback is stored in `feedback.csv` as shown below:

<img src="results/feedback_result_2.png" alt="Feedback Result 2"/>