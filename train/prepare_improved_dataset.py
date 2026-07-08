"""
改进数据集准备脚本
1. 合并低光照和白天数据集
2. 将 class 3 映射为 class 2 (Broken)
3. 生成暗光/雾霾模拟增强数据
4. 处理类别不平衡
"""
import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent  # ELF_论文增强数据目录
OUTPUT = BASE / "04-改进数据集"
LOW_LIGHT = BASE / "05-低光照数据集"

# ========== 配置 ==========
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 暗光/雾霾增强参数
DARK_FACTOR_RANGE = (0.3, 0.6)    # 亮度降低到30-60%
FOG_INTENSITY_RANGE = (0.3, 0.7)  # 雾气强度
NOISE_STD_RANGE = (5, 15)         # 高斯噪声标准差
AUGMENT_PROB = 0.3                # 每张图被增强的概率

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def apply_dark_augment(img):
    """模拟暗光场景"""
    factor = random.uniform(*DARK_FACTOR_RANGE)
    dark = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    # 添加少量噪声模拟低光照传感器噪声
    noise_std = random.uniform(*NOISE_STD_RANGE)
    noise = np.random.normal(0, noise_std, dark.shape).astype(np.int16)
    dark = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return dark


def apply_fog_augment(img):
    """模拟雾霾场景 - 大气散射模型"""
    h, w = img.shape[:2]
    # 生成雾气层
    intensity = random.uniform(*FOG_INTENSITY_RANGE)
    # 创建渐变的雾气（考虑景深）
    depth = random.uniform(0.5, 1.5)  # 雾气深度因子
    fog_layer = np.ones((h, w), dtype=np.float32) * intensity * 255
    
    # 添加 Perlin-like 随机性让雾更自然
    noise_scale = random.randint(20, 60)
    small_h, small_w = h // noise_scale + 1, w // noise_scale + 1
    noise = np.random.rand(small_h, small_w).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    fog_layer = fog_layer * (0.7 + 0.3 * noise)
    
    # 大气光
    atmosphere = random.uniform(0.85, 1.0)
    fog_layer = fog_layer * atmosphere
    
    fog_layer_3ch = np.stack([fog_layer] * 3, axis=2)
    foggy = np.clip(img.astype(np.float32) * (1 - intensity) + fog_layer_3ch * intensity, 0, 255).astype(np.uint8)
    return foggy


def apply_clahe_enhance(img):
    """CLAHE 自适应直方图均衡化 - 增强暗光对比度（用于验证集，不用于训练）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_labels(label_path, class_remap={3: 2}):
    """处理标注文件，重新映射类别"""
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id in class_remap:
                    parts[0] = str(class_remap[cls_id])
                # 排除未定义的 class 3（如果不需要它）
                if int(parts[0]) <= 2:  # 只保留 0,1,2
                    new_lines.append(' '.join(parts))
    return new_lines


def main():
    print("=" * 60)
    print("ELF 绝缘子检测 - 改进数据集准备")
    print("=" * 60)
    
    # 清理输出目录
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    
    # 创建目录结构
    for sub in ['images/train', 'images/val', 'images/test',
                 'labels/train', 'labels/val', 'labels/test']:
        (OUTPUT / sub).mkdir(parents=True, exist_ok=True)
    
    # ==================== 1. 收集所有图片 ====================
    all_images = {}  # basename -> (img_path, label_path)
    
    # 从低光照数据集收集（包含 class 3 标注，信息更丰富）
    for split in ['train', 'val']:
        lbl_dir = LOW_LIGHT / 'labels' / split
        img_dir = LOW_LIGHT / 'images' / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob('*.txt'):
            name = lbl_file.stem
            img_file = img_dir / f"{name}.jpg"
            if img_file.exists():
                all_images[name] = (str(img_file), str(lbl_file))
    
    print(f"\n总图片数: {len(all_images)}")
    
    # ==================== 2. 统计并处理类别 ====================
    total_class_counts = Counter()
    print("\n处理标注...")
    for name, (img_path, lbl_path) in all_images.items():
        new_lines = process_labels(lbl_path, class_remap={3: 2})
        for line in new_lines:
            cls_id = int(line.split()[0])
            total_class_counts[str(cls_id)] += 1
    
    print(f"处理前类别分布: {dict(total_class_counts)}")
    
    print(f"处理后类别分布(按名称):")
    for cls_id in sorted(total_class_counts.keys()):
        name_map = {'0': 'Normal(正常)', '1': 'JYZ(绝缘子)', '2': 'Broken(破损)'}
        print(f"  Class {cls_id} {name_map.get(cls_id, '?')}: {total_class_counts[cls_id]}")
    
    # ==================== 3. 划分 train/val/test ====================
    names = sorted(all_images.keys())
    random.shuffle(names)
    
    n_total = len(names)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_names = names[:n_train]
    val_names = names[n_train:n_train + n_val]
    test_names = names[n_train + n_val:]
    
    print(f"\n数据集划分:")
    print(f"  Train: {len(train_names)} 张 ({len(train_names)/n_total*100:.0f}%)")
    print(f"  Val:   {len(val_names)} 张 ({len(val_names)/n_total*100:.0f}%)")
    print(f"  Test:  {len(test_names)} 张 ({len(test_names)/n_total*100:.0f}%)")
    
    # ==================== 4. 复制/处理图片和标注 ====================
    def copy_to_split(name_list, split_name):
        img_out = OUTPUT / 'images' / split_name
        lbl_out = OUTPUT / 'labels' / split_name
        
        for name in name_list:
            img_path, lbl_path = all_images[name]
            shutil.copy2(img_path, img_out / f"{name}.jpg")
            
            new_lines = process_labels(lbl_path, class_remap={3: 2})
            with open(lbl_out / f"{name}.txt", 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
    
    print("\n复制训练集...")
    copy_to_split(train_names, 'train')
    print("复制验证集...")
    copy_to_split(val_names, 'val')
    print("复制测试集...")
    copy_to_split(test_names, 'test')
    
    # ==================== 5. 为训练集生成暗光/雾霾增强数据 ====================
    print("\n" + "=" * 50)
    print("生成暗光/雾霾增强训练数据...")
    print("=" * 50)
    
    img_train_dir = OUTPUT / 'images' / 'train'
    lbl_train_dir = OUTPUT / 'labels' / 'train'
    
    augmented_count = 0
    for img_file in sorted(img_train_dir.glob('*.jpg')):
        if random.random() > AUGMENT_PROB:
            continue
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        name = img_file.stem
        lbl_file = lbl_train_dir / f"{name}.txt"
        
        # --- 暗光增强 ---
        dark_img = apply_dark_augment(img)
        dark_name = f"{name}_DARK"
        cv2.imwrite(str(img_train_dir / f"{dark_name}.jpg"), dark_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(str(lbl_file), str(lbl_train_dir / f"{dark_name}.txt"))
        augmented_count += 1
        
        # --- 雾霾增强 ---
        fog_img = apply_fog_augment(img)
        fog_name = f"{name}_FOG"
        cv2.imwrite(str(img_train_dir / f"{fog_name}.jpg"), fog_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(str(lbl_file), str(lbl_train_dir / f"{fog_name}.txt"))
        augmented_count += 1
        
        # --- 极端暗光增强 ---
        if random.random() < 0.3:  # 30% of augmented images also get very dark version
            vdark_img = apply_dark_augment(img)
            vdark_img = np.clip(vdark_img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
            vdark_name = f"{name}_VDARK"
            cv2.imwrite(str(img_train_dir / f"{vdark_name}.jpg"), vdark_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(str(lbl_file), str(lbl_train_dir / f"{vdark_name}.txt"))
            augmented_count += 1
    
    print(f"生成了 {augmented_count} 张增强图片")
    
    # ==================== 6. 统计最终数据分布 ====================
    print("\n" + "=" * 50)
    print("最终数据集统计")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        img_dir = OUTPUT / 'images' / split
        lbl_dir = OUTPUT / 'labels' / split
        n_imgs = len(list(img_dir.glob('*.jpg')))
        
        cls_counts = Counter()
        for lbl_file in lbl_dir.glob('*.txt'):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_counts[parts[0]] += 1
        
        name_map = {'0': 'Normal', '1': 'JYZ', '2': 'Broken'}
        cls_str = ', '.join(f"{name_map.get(k,k)}={v}" for k,v in sorted(cls_counts.items()))
        print(f"  {split}: {n_imgs} imgs | {cls_str}")
    
    # ==================== 7. 生成 dataset.yaml ====================
    yaml_content = f"""# ELF 绝缘子检测 - 改进数据集
# 包含原图 + 暗光模拟 + 雾霾模拟增强
path: {str(OUTPUT.absolute()).replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test

# Classes
nc: 3
names:
  0: Normal
  1: JYZ
  2: Broken
"""
    
    yaml_path = OUTPUT / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n[OK] dataset.yaml 已生成: {yaml_path}")
    print(f"[OK] 数据集准备完成!")
    print(f"\n输出目录: {OUTPUT}")
    print(f"  - images/train: 训练图片（含暗光/雾霾增强）")
    print(f"  - images/val:   验证图片")
    print(f"  - images/test:  测试图片")
    print(f"  - labels/:      对应标注")
    print(f"  - dataset.yaml: YOLO数据集配置")


if __name__ == '__main__':
    main()
