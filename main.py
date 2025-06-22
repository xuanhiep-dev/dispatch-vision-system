import os
import cv2
import torch
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from torchvision import transforms
from modules.classifier import ClsModel
from modules.frame_processor import FrameProcessor


# Main app
def main():
    st.set_page_config(page_title="Vision Feedback System", layout="wide")
    st.title("Kitchen Dispatch Vision Feedback System")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det_model = YOLO('models/det_best.pt')
    tray_model, tray_classes = ClsModel.load(
        'models/cls_tray_best.pth', device=device)
    dish_model, dish_classes = ClsModel.load(
        'models/cls_dish_best.pth', device=device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # Define a frame processor
    frame_processor = FrameProcessor(
        transform, det_model, dish_model, dish_classes, tray_model, tray_classes)

    video_path = 'demo_videos/demo_video.mp4'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    cap.release()

    st.markdown("""
        <style>
        div[data-testid="stSlider"] {
            border: 2px solid #e7e5e5;
            padding: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]:nth-of-type(2)>div[data-testid="stVerticalBlock"] {
            overflow: auto;
            height: 75vh;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        left, right = st.columns([2, 1], border=True)
        with left:
            time_options = [frame_processor.seconds_to_hms(
                sec) for sec in range(int(total_duration) + 1)]

            selected_time_str = st.select_slider(
                "Select time:", options=time_options)

            h, m, s = map(int, selected_time_str.split(':'))
            selected_seconds = h * 3600 + m * 60 + s
            frame_idx = int(selected_seconds * fps)
            frame_idx = min(frame_idx, total_frames - 1)

            # Load frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            # Show image result
            if ret:
                processed_frame, detected_objects = frame_processor.process_frame_and_detect(
                    frame.copy())
                st.image(processed_frame, channels="BGR",
                         use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center; font-size:18px; color:black; font-weight:bold;'>"
                    f"Time {frame_processor.seconds_to_hms(selected_seconds)} | Frame {frame_idx}"
                    f"</p>", unsafe_allow_html=True
                )
            else:
                st.warning("Cannot read frame!")

        with right:
            if ret:
                for idx, obj in enumerate(detected_objects):
                    x1, y1, x2, y2 = obj['box']
                    crop_img = obj['crop']

                    st.write(f"### Object #{idx+1}")
                    st.image(cv2.cvtColor(
                        crop_img, cv2.COLOR_BGR2RGB), width=150)
                    st.write(
                        f"**Bounding Box:** ({x1}, {y1}), ({x2}, {y2})")
                    st.write(
                        f"**Detection label:** {obj['detection_label']} - **Classfication label:** {obj['classfication_label']}")
                    st.write(
                        f"**Confidence:** {obj['probability']*100:.2f}%")

                    detection_key = f"correct_detection_label_{idx}"
                    classification_key = f"correct_classification_label_{idx}"
                    detection_options = ['dish', 'tray']
                    classification_options = ['empty', 'not_empty', 'kakigori']

                    st.session_state.setdefault(
                        detection_key, obj['detection_label'])
                    st.session_state.setdefault(
                        classification_key, obj['classfication_label'])

                    selected_detection_label = st.selectbox(
                        "Correct detection class for this object:",
                        options=detection_options,
                        key=detection_key
                    )

                    selected_classification_label = st.selectbox(
                        "Correct classification label for this object:",
                        options=classification_options,
                        key=classification_key
                    )

                    if st.button(f"Save Feedback for Object #{idx+1}", key=f"save_feedback_{idx}"):
                        feedback_dir = "feedback_data"
                        os.makedirs(feedback_dir, exist_ok=True)
                        feedback_file = os.path.join(
                            feedback_dir, "feedback.csv")

                        # Create headlines
                        if not os.path.exists(feedback_file):
                            df = pd.DataFrame(columns=[
                                "time", "frame", "x1", "y1", "x2", "y2", "predicted_object_label", "corrected_object_label",
                                "predicted_class_label", "corrected_class_label", "probability"])
                        else:
                            df = pd.read_csv(feedback_file)

                        # Remove duplicat lines (same frame and coordinations)
                        duplicate_condition = (
                            (df["frame"] == frame_idx) &
                            (df["x1"] == x1) &
                            (df["y1"] == y1) &
                            (df["x2"] == x2) &
                            (df["y2"] == y2)
                        )
                        df = df[~duplicate_condition]

                        # Add a new line
                        new_row = {
                            "time": frame_processor.seconds_to_hms(selected_seconds),
                            "frame": frame_idx,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "predicted_object_label": obj['detection_label'],
                            "corrected_object_label": selected_detection_label,
                            "predicted_class_label": obj['classfication_label'],
                            "corrected_class_label": selected_classification_label,
                            "probability": obj['probability'],
                        }
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True)

                        # Recored the results
                        df.to_csv(feedback_file, index=False)

                        st.success(f"Feedback saved for Object #{idx+1}!")


if __name__ == "__main__":
    main()
