from ultralytics import YOLO

model = YOLO("/root/FDA-Net-main/yolo/ultralytics-main/ultralytics/output/contrast/yolo11n_DUO_YOLO/weights/best.pt")  # load a pretrained YOLOv8n model

model.predict(source="/root/lanyun-tmp/Dataset/domin2/images/train/2783.jpg",
               show=False,
               save = True) 
