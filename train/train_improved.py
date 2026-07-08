"""
ELF 绝缘子缺陷检测 - 完整训练脚本
- YOLOv8n 模型（适合部署到 RK3588）
- 改进数据集（含暗光/雾霾增强）
- 增强数据增强策略
- 处理类别不平衡
"""
import os
import sys
import argparse
from pathlib import Path

# 禁用 WandB 等在线服务
os.environ['WANDB_MODE'] = 'disabled'
os.environ['ULTRALYTICS_WANDB_DISABLE'] = '1'

# PyTorch 2.6+ 兼容性
import torch.serialization
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='ELF 绝缘子检测训练')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='模型路径 (yolov8n.pt / yolov8s.pt / 已有best.pt)')
    parser.add_argument('--data', type=str, 
                        default='../04-改进数据集/dataset.yaml',
                        help='dataset.yaml 路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入分辨率')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/0)')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--project', type=str, 
                        default='runs/improved_train',
                        help='输出目录')
    parser.add_argument('--name', type=str, default='insulator_v3',
                        help='实验名称')
    args = parser.parse_args()

    print("=" * 60)
    print("ELF 绝缘子缺陷检测 - 改进训练")
    print("=" * 60)
    print(f"  模型:        {args.model}")
    print(f"  数据集:      {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  分辨率:      {args.imgsz}")
    print(f"  Batch:       {args.batch}")
    print(f"  设备:        {args.device}")
    print(f"  早停:        {args.patience}")
    print(f"  学习率:      {args.lr0}")
    print("=" * 60)

    # 检查数据集
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集配置不存在: {data_path}")
        print("请先运行: python prepare_improved_dataset.py")
        sys.exit(1)

    # 加载模型
    model_path = Path(args.model)
    if model_path.exists():
        print(f"\n[INFO] 加载模型: {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"\n[INFO] 使用默认 yolov8n.pt")
        model = YOLO('yolov8n.pt')

    # 训练参数
    train_args = {
        'data': str(data_path.absolute()),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'lr0': args.lr0,
        'lrf': 0.01,              # 最终学习率 = lr0 * lrf
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        
        # === 优化器 ===
        'optimizer': 'AdamW',     # AdamW 对类别不平衡更鲁棒
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # === 数据增强 ===
        'hsv_h': 0.015,           # 色调抖动
        'hsv_s': 0.7,             # 饱和度抖动（大幅变化模拟不同光照）
        'hsv_v': 0.5,             # 明度抖动（覆盖暗光场景）
        'degrees': 0.0,           # 不旋转（绝缘子有方向性）
        'translate': 0.1,         # 平移
        'scale': 0.5,             # 缩放（多尺度训练）
        'shear': 2.0,             # 轻微剪切（模拟不同角度拍摄）
        'perspective': 0.0,       # 不透视变换
        'flipud': 0.0,            # 不上下翻转
        'fliplr': 0.5,            # 50%概率左右翻转
        
        # === 组合增强 ===
        'mosaic': 1.0,            # 100%马赛克增强
        'mixup': 0.2,             # 20% MixUp（帮助类别平衡）
        'copy_paste': 0.1,        # 10% Copy-Paste（增加小目标样本）
        'close_mosaic': 15,       # 最后15轮关闭mosaic
        
        # === 其他 ===
        'label_smoothing': 0.1,   # 标签平滑（防止过拟合）
        'dropout': 0.1,           # 分类头 dropout
        'cos_lr': True,           # 余弦退火学习率
        'warmup_epochs': 3,       # 预热轮数
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # === 保存 ===
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': False,
        
        # === 性能 ===
        'amp': False,             # CPU不支持AMP
        'rect': False,            # 不矩形训练（画面比例多样）
    }

    print(f"\n[INFO] 开始训练...")
    print(f"  训练样本: {args.imgsz}x{args.imgsz}, batch={args.batch}")
    
    try:
        results = model.train(**train_args)
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        # 找到最佳模型
        best_path = Path(args.project) / args.name / 'weights' / 'best.pt'
        if best_path.exists():
            import shutil
            deploy_path = Path(r'best_improved.pt')
            shutil.copy2(str(best_path), str(deploy_path))
            print(f"\n[OK] 最佳模型已保存到: {deploy_path}")
            print(f"  模型大小: {best_path.stat().st_size / 1024**2:.1f} MB")
        
        return results
        
    except KeyboardInterrupt:
        print("\n[INFO] 训练被中断")
    except Exception as e:
        print(f"\n[ERROR] 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
