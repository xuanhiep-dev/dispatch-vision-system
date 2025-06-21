import cv2
import torch
import torch.nn.functional as F
import base64
from PIL import Image
from io import BytesIO


class FrameProcessor:
    def __init__(self, transform, det_model, dish_model, dish_classes, tray_model, tray_classes, threshold=0.80, alpha=0.3):
        self.transform = transform
        self.det_model = det_model
        self.dish_model = dish_model
        self.dish_classes = dish_classes
        self.tray_model = tray_model
        self.tray_classes = tray_classes
        self.threshold = threshold
        self.alpha = alpha

        self.display_colors = {
            'empty': (255, 120, 0),
            'not_empty': (0, 130, 255),
            'kakigori': (255, 110, 255)
        }
        self.display_labels = {
            'empty': 'empty',
            'not_empty': 'not empty',
            'kakigori': 'kakigori'
        }

    @staticmethod
    def seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def image_to_base64(img_array):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode()
        return base64_str

    def draw_box_with_label(self, frame, box, box_class, class_name):
        display_label = self.display_labels.get(class_name, class_name)
        display_label = f"{str(box_class)}: {str(display_label)}"
        color = self.display_colors.get(class_name, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        text_size = cv2.getTextSize(
            display_label, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 20 else y2 + text_size[1] + 10
        cv2.rectangle(frame, (text_x, text_y -
                      text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1)
        cv2.putText(frame, display_label, (text_x + 2, text_y),
                    font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, display_label, (text_x + 2, text_y),
                    font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    def process_frame_and_detect(self, frame):
        results = self.det_model(frame, imgsz=1312, augment=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        valid_idx = scores >= self.threshold
        boxes = boxes[valid_idx]
        class_ids = class_ids[valid_idx]
        scores = scores[valid_idx]

        detected_objects = []

        dish_crops, dish_coords, dish_crops_images = [], [], []
        tray_crops, tray_coords, tray_crops_images = [], [], []

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            crop_tensor = self.transform(crop).unsqueeze(0)
            if class_id == 0:  # dish
                box_class = "dish"
                dish_crops.append(crop_tensor)
                dish_coords.append((x1, y1, x2, y2))
                dish_crops_images.append(crop)
            else:  # tray
                box_class = "tray"
                tray_crops.append(crop_tensor)
                tray_coords.append((x1, y1, x2, y2))
                tray_crops_images.append(crop)

        if dish_crops:
            dish_batch = torch.cat(dish_crops, dim=0)
            dish_outputs = self.dish_model(dish_batch)
            probs = F.softmax(dish_outputs, dim=1)
            dish_probs, dish_preds = torch.max(probs, dim=1)
            dish_probs = dish_probs.detach().cpu().numpy()
            dish_preds = dish_preds.detach().cpu().numpy()
            for (x1, y1, x2, y2), pred_class, prob, crop_img in zip(dish_coords, dish_preds, dish_probs, dish_crops_images):
                class_name = self.dish_classes[pred_class]
                self.draw_box_with_label(
                    frame, (x1, y1, x2, y2), box_class, class_name)
                detected_objects.append({
                    'box': (x1, y1, x2, y2),
                    'detection_label': box_class,
                    'classfication_label': class_name,
                    'probability': float(prob),
                    'crop': crop_img
                })

        if tray_crops:
            tray_batch = torch.cat(tray_crops, dim=0)
            tray_outputs = self.tray_model(tray_batch)
            probs = F.softmax(tray_outputs, dim=1)
            tray_probs, tray_preds = torch.max(probs, dim=1)
            tray_probs = tray_probs.detach().cpu().numpy()
            tray_preds = tray_preds.detach().cpu().numpy()
            for (x1, y1, x2, y2), pred_class, prob, crop_img in zip(tray_coords, tray_preds, tray_probs, tray_crops_images):
                class_name = self.tray_classes[pred_class]
                self.draw_box_with_label(
                    frame, (x1, y1, x2, y2), box_class, class_name)
                detected_objects.append({
                    'box': (x1, y1, x2, y2),
                    'detection_label': box_class,
                    'classfication_label': class_name,
                    'probability': float(prob),
                    'crop': crop_img
                })

        return frame, detected_objects
