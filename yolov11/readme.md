This system achieves precise detection of six object categories—monitor (patient monitor), staff (medical personnel), IV_stand (IV stand), pump (infusion pump), vent (ventilator), and bed (hospital bed)—using a custom-trained YOLOv11 model. By integrating Depth Anything v2 for depth estimation, the system ultimately generates 3D bounding boxes and bird's-eye view (BEV) visualizations.

./train

  Construct a medical-scene-specific dataset, including annotated data for six target categories such as monitor and staff; Train a custom object detection model based on YOLOv11 to optimize the detection accuracy of medical targets.
  Prepare dateset and Customize model based on yollov11 is stored in weights/best.pt

==================================================
  Refer to custom.yaml to prepare dataset

    数据集路径
    train:
    
    ./dataset/train/images # 训练集图片路径
    ./dataset/train/negative_images val: ./dataset/val/images # 验证集图片路径 test: ./dataset/test/images # 新增测试集图片路径
    数据集路径
    labels: train: - ./dataset/train/labels # 训练集图片路径 - ./dataset/train/negative_labels val: ./dataset/val/labels # 验证集图片路径 test: ./dataset/test/labels # 新增测试集图片路径
    
    类别数量
    nc: 6
    
    类别名称
    names: [ 'monitor', 'bed','IV_stand','vent','pump','person' ]
    task: detect

===============================================

