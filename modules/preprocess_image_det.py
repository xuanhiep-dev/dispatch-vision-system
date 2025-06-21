import os
import cv2
import numpy as np
import albumentations as A
import shutil


class YOLODatasetAugmentor:
    def __init__(self,
                 input_train_images,
                 input_train_labels,
                 input_val_images,
                 input_val_labels,
                 output_train_images,
                 output_train_labels,
                 output_val_images,
                 output_val_labels,
                 augmentations_per_image=5):

        self.input_train_images = input_train_images
        self.input_train_labels = input_train_labels
        self.input_val_images = input_val_images
        self.input_val_labels = input_val_labels
        self.output_train_images = output_train_images
        self.output_train_labels = output_train_labels
        self.output_val_images = output_val_images
        self.output_val_labels = output_val_labels
        self.augmentations_per_image = augmentations_per_image

        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussNoise(p=0.4),
            A.ImageCompression(p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def prepare_output_dirs(self):
        for path in [self.output_train_images, self.output_train_labels, self.output_val_images, self.output_val_labels]:
            os.makedirs(path, exist_ok=True)

    def copy_validation_data(self):
        for file in os.listdir(self.input_val_images):
            shutil.copy(os.path.join(self.input_val_images, file),
                        os.path.join(self.output_val_images, file))
        for file in os.listdir(self.input_val_labels):
            shutil.copy(os.path.join(self.input_val_labels, file),
                        os.path.join(self.output_val_labels, file))

    def augment_training_data(self):
        for img_file in os.listdir(self.input_train_images):
            img_path = os.path.join(self.input_train_images, img_file)
            label_path = os.path.join(
                self.input_train_labels, img_file.replace('.jpg', '.txt'))

            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image {img_file}")
                continue

            bboxes, class_labels = self.read_labels(label_path)

            # Copy original image and label
            shutil.copy(img_path, os.path.join(
                self.output_train_images, img_file))
            shutil.copy(label_path, os.path.join(
                self.output_train_labels, img_file.replace('.jpg', '.txt')))

            for i in range(self.augmentations_per_image):
                try:
                    augmented = self.transform(
                        image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    aug_img_name = img_file.replace('.jpg', f'_aug{i}.jpg')
                    cv2.imwrite(os.path.join(
                        self.output_train_images, aug_img_name), aug_img)

                    with open(os.path.join(self.output_train_labels, aug_img_name.replace('.jpg', '.txt')), 'w') as f:
                        for bbox, cls in zip(aug_bboxes, aug_labels):
                            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
                except Exception as e:
                    print(f"Error augmenting {img_file}: {e}")

    def read_labels(self, label_path):
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(cls)
        return bboxes, class_labels

    def run(self):
        self.prepare_output_dirs()
        self.copy_validation_data()
        self.augment_training_data()
        print("Augmentation completed.")


# ================== Example usage ==================
if __name__ == "__main__":
    augmentor = YOLODatasetAugmentor(
        input_train_images='Dataset/Detection/train/images',
        input_train_labels='Dataset/Detection/train/labels',
        input_val_images='Dataset/Detection/val/images',
        input_val_labels='Dataset/Detection/val/labels',
        output_train_images='Dataset/Detection_augmented/train/images',
        output_train_labels='Dataset/Detection_augmented/train/labels',
        output_val_images='Dataset/Detection_augmented/val/images',
        output_val_labels='Dataset/Detection_augmented/val/labels',
        augmentations_per_image=5
    )

    augmentor.run()
