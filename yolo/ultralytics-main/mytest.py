from ultralytics import YOLO, DuModel

# Load a model
model = YOLO("/root/lanyun-tmp/yolo/yolo12s_duo/weights/best.pt")

# Validate the model

metrics = model.val(data="/root/FDA-Net-main/yolo/ultralytics-main/ultralytics/cfg/datasets/underwater_DUO.yaml")  

metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category