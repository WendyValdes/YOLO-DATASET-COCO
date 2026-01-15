YOLO Object Detection — COCO (Person & Car)
Project Description

This project applies YOLOv8 object detection using a filtered subset of the COCO 2017 dataset, containing only the classes person and car. The goal is to practice dataset selection, conversion to YOLO format, and transfer learning with a pre-trained model.


- Dataset

Source: COCO 2017

Classes used: person and car

Samples:

  Training: 500 images

  Validation: 100 images

The dataset was exported to YOLO format using FiftyOne.


- Technologies

Python

FiftyOne

Ultralytics YOLOv8

Google Colab


-- How to Run

Install dependencies:

!pip install -U fiftyone ultralytics


Load and filter the dataset, export it to YOLO format, and train the model with:

model = YOLO("yolov8n.pt")

model.train(
    data=dataset_yaml_path,
    epochs=10,
    imgsz=512,
    batch=16,
    device=0
)


- Inference

Test the model on new images:

model = YOLO("/content/runs/detect/train2/weights/best.pt")
model.predict(source="test.jpg", conf=0.25, save=True)


- Results are saved in:

/content/runs/detect/predict/


- Results

The model successfully detects people and cars in most images. Performance can be improved with more data, more epochs, or a larger YOLO model (e.g., yolov8s.pt).
