from ultralytics import DuModel

if __name__ == "__main__":

    model = DuModel("/root/FDA-Net-main/yolo/ultralytics-main/ultralytics/cfg/models/11/yolo11s-allsae.yaml") 
    
    model.train(
        data="/root/FDA-Net-main/yolo/ultralytics-main/ultralytics/cfg/datasets/DG_6t1.yaml",  

        
        epochs=400,          
        batch=16,            
        optimizer="SGD",     
        lr0=0.01,            
        seed=0,              
        imgsz=640,           
        workers=8,          
        device=0,            
        pretrained=False,  
        
        project="/root/lanyun-tmp/yolo/", 
        name="sae", 
        save_json=True,  
    )