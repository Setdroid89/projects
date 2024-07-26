import cv2
import os
import torch

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Cambia a 1 si estás usando la cámara trasera

# Variables para controlar la captura
capturing = False
capture_duration = 3  # Duración de la captura en segundos
frames = []

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

        # Mostrar el resultado completo (sin recortar)
        cv2.imshow('YOLO', frame)

        # Capturar imágenes cuando se presione la tecla "s"
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            capturing = True
            frames.append(cropped_area)

    # Detener la captura después de la duración especificada
    if capturing and len(frames) >= capture_duration * 30:  # 30 FPS
        break

cap.release()
cv2.destroyAllWindows()

# Guardar solo la imagen de la patente
if frames:
    cv2.imwrite("captured_patente.png", frames[-1])

print("Imagen de la patente guardada como captured_patente.png")
