import cv2
import torch
import supervision as sv
from ultralytics import YOLOv10
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image, ImageDraw, ImageFont
import numpy as np

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
else:
    print("No GPUs available.")

# Additionally, to check for CPU availability:
cpu_available = True
print(f"CPU available: {cpu_available}")

# Ensure GPU availability and compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mewnames = {
    0: 'new_name_0',
    1: 'new_name_1',
    2: 'new_name_10',
    3: 'new_name_11',
    4: 'new_name_12',
    5: 'new_name_13',
    6: 'new_name_14',
    7: 'new_name_15',
    8: 'new_name_16',
    9: 'new_name_17',
    10: 'new_name_18',
    11: 'new_name_19',
    12: 'new_name_2',
    13: 'new_name_20',
    14: 'new_name_21',
    15: 'new_name_22',
    16: 'new_name_23',
    17: 'new_name_24',
    18: 'new_name_25',
    19: 'new_name_26',
    20: 'new_name_27',
    21: 'new_name_28',
    22: 'new_name_29',
    23: 'new_name_3',
    24: 'new_name_30',
    25: 'new_name_4',
    26: 'new_name_5',
    27: 'new_name_6',
    28: 'new_name_7',
    29: 'new_name_8',
    30: 'new_name_9',
    31: 'انا',
    32: 'احبك',
    33: 'عمري',
    34: 'اسمي'
}

model = YOLOv10(f'bestn.pt', names=mewnames).to(device)

print(model.model.names)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
else:
    print("Webcam opened successfully.")

# Load the TTF font
font_path = "Janna LT Bold.ttf"  # Path to your TTF font file
font = ImageFont.truetype(font_path, 20)

img_counter = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    
    # Convert frame to PIL image
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_image_pil)
    
    # Annotate labels
    for detection in detections:
        bbox = detection.bbox
        label = detection.label
        
        # Draw text with PIL
        draw.text((bbox[0], bbox[1]), label, font=font, fill=(255, 0, 0))
    
    # Convert back to OpenCV image
    annotated_image = cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('webcamera', annotated_image)
    
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
