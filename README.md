# Installation & Training Guide
## 1. Install.
```bash
pip install -r requirements.txt
```
## 2. Training.
### The training configuration is located in the tray_dish_detection_classification.ipynb file. With two steps:
1. Train an object detection model to identify trays and dishes.
2. Train a classification model to categorize the detected objects into three classes: "not_empty", "empty", and "kakigori".
After training, three models are generated: best.pth (detection model), cls_dish_best.pth, and cls_tray_best.pth (classification models).

# 3. Evaluation on the validation dataset.
### 3.1. The valid detection result.
![Detection Result](results/detection_result.png)
### 3.2. The classification results.
1. Tray object: 96.68 (Best Val Acc)
2. Dish object: 96.46 (Best Val Acc)

## 4. Inference.
### 4.1. Load streamlit tool.
The project is developed using the Streamlit tool. To execute the project:
```bash
streamlit run main.py
```
### 4.2. Get the project output.
### The project interface has several main features:
1. The left column displays the prediction results in image format, with bounding boxes and corresponding object labels.
2. The right column presents detailed information, including bounding box coordinates, detection labels, and classification labels.
3. Additionally, this section includes a user feedback feature. Users can provide corrections for both detection and classification labels, and the feedback will be saved in a CSV file to support model performance improvement.
4. The feedback file will be named feedback.csv and stored in the feedback_data directory. This file will record essential information such as: detection timestamp, specific frame index, object coordinates, predicted labels, user feedback labels for both detection and classification tasks, and the prediction confidence score.
#### The picture below shows an example of a project result.
![Project Result](results/project_result.png)

#### The picture below shows an example of a feedback result.
<img src="results/feedback_result.png" alt="Feedback Result" width="350"/>
