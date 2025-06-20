import torch.nn.functional as F
import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from modules.classifier import ClsModel
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os

# Draw function


def draw_box_with_label(frame, box, class_name, alpha=0.3):
    display_colors = {'empty': (255, 120, 0), 'not_empty': (
        0, 130, 255), 'kakigori': (255, 110, 255)}
    display_labels = {'empty': 'Empty',
                      'not_empty': 'Not Empty', 'kakigori': 'Kakigori'}
    display_label = display_labels[class_name]
    color = display_colors[class_name]
    x1, y1, x2, y2 = map(int, box)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_size = cv2.getTextSize(display_label, font, font_scale, thickness)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 20 else y2 + text_size[1] + 10
    cv2.rectangle(frame, (text_x, text_y -
                  text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1)
    cv2.putText(frame, display_label, (text_x + 2, text_y),
                font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, display_label, (text_x + 2, text_y), font,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_frame_and_detect(frame, transform, det_model, dish_model, dish_classes, tray_model, tray_classes):
    results = det_model(frame, imgsz=1312, augment=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    scores = results[0].boxes.conf.cpu().numpy()

    # Chỉ lấy box có confidence >= 85%
    threshold = 0.80
    valid_idx = scores >= threshold

    boxes = boxes[valid_idx]
    class_ids = class_ids[valid_idx]
    scores = scores[valid_idx]

    detected_objects = []

    dish_crops, dish_coords = [], []
    tray_crops, tray_coords = [], []
    dish_crops_images, tray_crops_images = [], []

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        crop_tensor = transform(crop).unsqueeze(0)
        if class_id == 0:  # dish
            dish_crops.append(crop_tensor)
            dish_coords.append((x1, y1, x2, y2))
            dish_crops_images.append(crop)
        else:  # tray
            tray_crops.append(crop_tensor)
            tray_coords.append((x1, y1, x2, y2))
            tray_crops_images.append(crop)

    if dish_crops:
        dish_batch = torch.cat(dish_crops, dim=0)
        dish_outputs = dish_model(dish_batch)
        probs = F.softmax(dish_outputs, dim=1)
        dish_probs, dish_preds = torch.max(probs, dim=1)
        dish_probs = dish_probs.detach().cpu().numpy()
        dish_preds = dish_preds.detach().cpu().numpy()
        for (x1, y1, x2, y2), pred_class, prob, crop_img in zip(dish_coords, dish_preds, dish_probs, dish_crops_images):
            class_name = dish_classes[pred_class]
            draw_box_with_label(frame, (x1, y1, x2, y2), class_name)
            detected_objects.append({
                'box': (x1, y1, x2, y2),
                'predicted_label': class_name,
                'probability': float(prob),
                'crop': crop_img
            })

    if tray_crops:
        tray_batch = torch.cat(tray_crops, dim=0)
        tray_outputs = tray_model(tray_batch)
        probs = F.softmax(tray_outputs, dim=1)
        tray_probs, tray_preds = torch.max(probs, dim=1)
        tray_probs = tray_probs.detach().cpu().numpy()
        tray_preds = tray_preds.detach().cpu().numpy()
        for (x1, y1, x2, y2), pred_class, prob, crop_img in zip(tray_coords, tray_preds, tray_probs, tray_crops_images):
            class_name = tray_classes[pred_class]
            draw_box_with_label(frame, (x1, y1, x2, y2), class_name)
            detected_objects.append({
                'box': (x1, y1, x2, y2),
                'predicted_label': class_name,
                'probability': float(prob),
                'crop': crop_img
            })

    return frame, detected_objects


def image_to_base64(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return base64_str

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

    video_path = 'demo_videos/demo_video.mp4'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    cap.release()

    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = 0.0

    selected_seconds = st.slider("Select time:", 0.0, total_duration,
                                 st.session_state.selected_time, step=1.0, format="%.0f")

    frame_idx = int(selected_seconds * fps)
    frame_idx = min(frame_idx, total_frames - 1)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    left, right = st.columns([2, 1])

    if ret:
        processed_frame, detected_objects = process_frame_and_detect(frame.copy(
        ), transform, det_model, dish_model, dish_classes, tray_model, tray_classes)

        with left:
            st.image(processed_frame, channels="BGR", use_column_width=True)
            st.markdown(
                f"<p style='text-align:center; font-size:18px; color:black; font-weight:bold;'>"
                f"Time {seconds_to_hms(selected_seconds)} | Frame {frame_idx}"
                f"</p>", unsafe_allow_html=True
            )

        with right:
            scroll_container = st.container()
            with scroll_container:
                st.markdown("""
                    <style>
                    div[data-testid="column"]:nth-of-type(2) > div {
                        height: 68vh;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        padding: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                for idx, obj in enumerate(detected_objects):
                    x1, y1, x2, y2 = obj['box']
                    crop_img = obj['crop']

                    st.write(f"### Object #{idx+1}")
                    st.image(cv2.cvtColor(
                        crop_img, cv2.COLOR_BGR2RGB), width=150)
                    st.write(f"**Bounding Box:** ({x1}, {y1}), ({x2}, {y2})")
                    st.write(f"**Predicted label:** {obj['predicted_label']}")
                    st.write(f"**Confidence:** {obj['probability']*100:.2f}%")

                    label_options = ['empty', 'not_empty', 'kakigori']
                    selected_label = st.selectbox(
                        "Correct label:",
                        options=label_options,
                        index=label_options.index(obj['predicted_label']),
                        key=f"correct_label_{idx}"
                    )

                    if st.button(f"Save Feedback for Object #{idx+1}", key=f"save_feedback_{idx}"):
                        import pandas as pd

                        feedback_dir = "feedback_data"
                        os.makedirs(feedback_dir, exist_ok=True)
                        feedback_file = os.path.join(
                            feedback_dir, "feedback.csv")

                        # Nếu file chưa tồn tại thì tạo file mới kèm header
                        if not os.path.exists(feedback_file):
                            df = pd.DataFrame(columns=[
                                "time", "frame", "x1", "y1", "x2", "y2",
                                "predicted_label", "probability", "corrected_label"
                            ])
                        else:
                            df = pd.read_csv(feedback_file)

                        # Xoá các dòng trùng (cùng frame và cùng tọa độ)
                        duplicate_condition = (
                            (df["frame"] == frame_idx) &
                            (df["x1"] == x1) &
                            (df["y1"] == y1) &
                            (df["x2"] == x2) &
                            (df["y2"] == y2)
                        )
                        df = df[~duplicate_condition]

                        # Thêm dòng mới
                        new_row = {
                            "time": seconds_to_hms(selected_seconds),
                            "frame": frame_idx,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "predicted_label": obj['predicted_label'],
                            "probability": obj['probability'],
                            "corrected_label": selected_label
                        }
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True)

                        # Ghi lại toàn bộ file
                        df.to_csv(feedback_file, index=False)

                        st.success(f"✅ Feedback saved for Object #{idx+1}!")

                    st.markdown("---")

    else:
        st.warning("Cannot read frame!")


if __name__ == "__main__":
    main()
