# predict.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import json
import pandas as pd
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Config
num_classes = 11
score_thresh = 0.5

test_img_folder = '../nycu-hw2-data/nycu-hw2-data/test'
model_path = 'fasterrcnn_digit_model_5.pth'

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().cuda()

log_file = open("pred_output.txt", "w")

transform = transforms.ToTensor()
pred_json = []
pred_csv = []
image_files = sorted(os.listdir(test_img_folder))

for image_name in image_files:
    image_path = os.path.join(test_img_folder, image_name)
    image = transform(Image.open(image_path).convert("RGB")).cuda()

    with torch.no_grad():
        output = model([image])[0]

    boxes = output['boxes'].cpu().tolist()
    labels = output['labels'].cpu().tolist()
    scores = output['scores'].cpu().tolist()

    image_id = int(os.path.splitext(image_name)[0])
    detected_digits = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thresh:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            cat_id = label

            pred_json.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x_min, y_min, width, height],
                "score": score
            })

            detected_digits.append((x_min, str(label - 1)))
        #print(f'{x_min:.4f}, {y_min:.4f}, {width:.4f}, {height:.4f}')

    if detected_digits:
        detected_digits.sort(key=lambda x: x[0])  # Sort by x position
        pred_label = ''.join([d[1] for d in detected_digits])
    else:
        pred_label = -1

    print(f'{image_id}: {pred_label}')
    log_file.write(f"{image_id}: {pred_label}\n")

    pred_csv.append({"image_id": image_id, "pred_label": pred_label})

log_file.close()

# Save to JSON file
with open("pred.json", "w") as f:
    json.dump(pred_json, f)

# Save pred.csv
pd.DataFrame(pred_csv).to_csv('pred.csv', index=False)

print("all files are saved")
