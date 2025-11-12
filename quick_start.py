"""
快速开始脚本 - 创建示例数据集结构
Quick start script - Create example dataset structure
"""
import os
from pathlib import Path


def create_dataset_structure():
    """创建数据集目录结构 / Create dataset directory structure"""
    
    # 创建目录 / Create directories
    dirs = [
        "dataset/images",
        "dataset/labels",
        "checkpoints",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录 / Created directory: {dir_path}")
    
    # 创建示例标签文件 / Create example label file
    example_label = """# 标签格式示例 / Label format example
# 每行格式 / Each line format: class_id x_center y_center width height
# 所有坐标都归一化到 [0, 1] / All coordinates normalized to [0, 1]

# 示例 / Examples:
6 0.535577 0.515385 0.234615 0.230769
6 0.350962 0.415385 0.419231 0.296154
6 0.751923 0.546154 0.105769 0.334615

# 坐标说明 / Coordinate explanation:
# class_id: 类别编号 (0, 1, 2, ...) / Class ID (0, 1, 2, ...)
# x_center: 边界框中心x坐标 / Bounding box center x coordinate
# y_center: 边界框中心y坐标 / Bounding box center y coordinate
# width: 边界框宽度 / Bounding box width
# height: 边界框高度 / Bounding box height

# 注意：删除所有注释行后使用！
# Note: Remove all comment lines before use!
"""
    
    with open("dataset/labels/example_format.txt", "w") as f:
        f.write(example_label)
    print("✓ 创建示例标签文件 / Created example label file: dataset/labels/example_format.txt")
    
    # 创建README / Create README
    readme = """# 数据集目录 / Dataset Directory

## 使用说明 / Instructions

1. **添加图片 / Add Images**
   - 将你的PNG图片放到 `images/` 目录
   - Place your PNG images in the `images/` directory

2. **添加标签 / Add Labels**
   - 在 `labels/` 目录创建对应的TXT文件
   - Create corresponding TXT files in the `labels/` directory
   - 文件名必须与图片名相同（不含扩展名）
   - Filename must match the image name (without extension)
   
3. **标签格式 / Label Format**
   - 查看 `example_format.txt` 了解格式
   - Check `example_format.txt` for format reference
   - 每行一个目标: class_id x y w h
   - One object per line: class_id x y w h
   - 所有值归一化到 [0, 1]
   - All values normalized to [0, 1]

## 示例 / Example

如果你有一张图片 `defect_001.png`:
If you have an image `defect_001.png`:

1. 将图片放到: `images/defect_001.png`
   Place image at: `images/defect_001.png`

2. 创建标签: `labels/defect_001.txt`
   Create label: `labels/defect_001.txt`

3. 标签内容:
   Label content:
   ```
   0 0.5 0.5 0.3 0.4
   1 0.7 0.3 0.2 0.2
   ```
"""
    
    with open("dataset/README.txt", "w") as f:
        f.write(readme)
    print("✓ 创建README文件 / Created README file: dataset/README.txt")
    
    print("\n" + "=" * 60)
    print("数据集目录结构创建完成！/ Dataset structure created!")
    print("=" * 60)
    print("\n下一步 / Next steps:")
    print("1. 将PNG图片放到 dataset/images/")
    print("   Put PNG images in dataset/images/")
    print("2. 将标签TXT放到 dataset/labels/")
    print("   Put label TXT files in dataset/labels/")
    print("3. 运行训练: python train_example.py")
    print("   Run training: python train_example.py")
    print("\n详细说明请查看: USAGE_GUIDE_CN.md")
    print("See detailed guide: USAGE_GUIDE_CN.md")


def check_environment():
    """检查Python环境 / Check Python environment"""
    print("检查Python环境 / Checking Python environment...")
    print("=" * 60)
    
    # 检查Python版本 / Check Python version
    import sys
    print(f"Python版本 / Python version: {sys.version}")
    
    # 检查必需的包 / Check required packages
    packages = {
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} 已安装 / installed")
        except ImportError:
            print(f"✗ {name} 未安装 / not installed")
            missing.append(name)
    
    if missing:
        print("\n需要安装以下包 / Need to install:")
        print("pip install torch torchvision pillow numpy matplotlib")
    else:
        print("\n✓ 所有依赖已安装 / All dependencies installed")
    
    print("=" * 60)


def show_file_summary():
    """显示文件说明 / Show file summary"""
    print("\n" + "=" * 60)
    print("文件说明 / File Summary")
    print("=" * 60)
    
    files = {
        "train_example.py": "训练脚本 / Training script",
        "inference_example.py": "推理脚本 / Inference script",
        "test_yolo_flow.py": "单元测试 / Unit tests",
        "USAGE_GUIDE_CN.md": "详细使用指南 / Detailed usage guide",
        "README.md": "项目说明 / Project overview",
        "yolov11.py": "模型定义 / Model definition",
        "criterion/detect_loss.py": "损失函数 / Loss function",
    }
    
    for filename, description in files.items():
        status = "✓" if Path(filename).exists() else "✗"
        print(f"{status} {filename:30s} - {description}")
    
    print("=" * 60)


def main():
    print("\n")
    print("=" * 60)
    print("YOLOv11 快速开始 / Quick Start")
    print("=" * 60)
    print()
    
    # 检查环境 / Check environment
    check_environment()
    
    # 创建数据集结构 / Create dataset structure
    print()
    create_dataset_structure()
    
    # 显示文件说明 / Show file summary
    show_file_summary()
    
    print("\n✓ 设置完成！准备开始训练 / Setup complete! Ready to train")
    print()


if __name__ == "__main__":
    main()
