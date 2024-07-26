import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Cambia a 1 si estás usando la cámara trasera

while cap.isOpened():
    ret, frame = cap.read()

    # Realizar detecciones
    results = model(frame)

    # Obtener coordenadas de los bounding boxes
    boxes = results.pred[0][:, :4].cpu().numpy()

    # Recortar y procesar cada bounding box
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cropped_area = frame[y1:y2, x1:x2]

        # Aquí puedes procesar la imagen recortada (por ejemplo, reconocimiento de caracteres)

        # Mostrar el resultado
        cv2.imshow('YOLO', cropped_area)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
