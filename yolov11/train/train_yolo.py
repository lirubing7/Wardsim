from ultralytics import YOLO
import os

# 设置中文显示（如果需要可视化）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载YOLO11预训练模型（使用nano版本作为基础）
#model = YOLO('yolo11n.pt')
model = YOLO('yolo11s.pt')

# 开始训练
results = model.train(
    data='custom.yaml',  # 配置文件路径
    epochs=100,           # 训练轮次，可根据效果调整
    batch=16,            # 批次大小，根据GPU内存调整
    imgsz=640,           # 输入图片尺寸
    device=0,            # 使用GPU训练，0表示第一块GPU，cpu表示使用CPU
    workers=4,           # 数据加载线程数
    project='hospital_detection',  # 项目名称
    name='yolo11_monitor_bed',     # 训练任务名称
    pretrained=True,     # 使用预训练权重
    optimizer='Adam',    # 优化器
    lr0=0.0005,          # 初始学习率
    patience=30,         # 早停耐心值
    save=True,           # 保存模型
    augment=True,        # 开启强数据增强 
#    model=yolov11s.pt,       # 换用稍大模型
#    class_weights=[2.0, 1.0],# 平衡monitor和bed的损失权重
#    conf=0.3,
    cache=True           # 缓存数据加速训练
)

# 训练完成后在验证集上评估
metrics = model.val()
print("验证集评估结果：")
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

# 保存最终模型
final_model_path = os.path.join('weights', 'best.pt')
print(f"最佳模型已保存至: {final_model_path}")
    
