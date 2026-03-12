# Solar Panel Health Monitoring & Defect Detection

This project leverages deep learning to automate the inspection of solar panels, identifying dust accumulation that reduces energy efficiency. It features a complete pipeline from model training (using transfer learning) to real-time video inference with dynamic Region of Interest (ROI) tracking.

## Features

- **Multi-Architecture Support**: Compare performance across VGG16, ResNet50, InceptionV3, and DenseNet121.
- **Dynamic ROI Tracking**: Automated localization of solar panels using Canny edge detection and contour analysis.
- **Real-Time Inference**: Processes video streams with classification overlays ("Clean" vs. "Dusty").
- **Slow-Motion Analysis**: Support for custom slowdown factors (e.g., 0.5x) to analyze impairment transitions.
- **Transfer Learning**: Optimized weights via two-stage training (feature extraction + fine-tuning).

## Performance Summary

The following metrics were achieved on the validation dataset (512 images) after fine-tuning on a GPU-accelerated environment (RTX 4060).

| Model | Test Accuracy | Precision (Dusty) | Inference Speed | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **VGG16** | **86%** | **87%** | 15 FPS | Best for high-precision offline audits. |
| **ResNet50** | **86%** | 85% | **24 FPS** | Best balance for real-time monitoring. |
| **InceptionV3** | 85% | 84% | 20 FPS | Robust multi-scale feature capture. |
| **DenseNet121** | 84% | 81% | 22 FPS | Highest recall for early dust detection. |

## 🛠 Project Structure

- `solar_panel_video_analysis.py`: Main script for model building and real-time video processing.
- `SolarPanel_CaseStudy_*.ipynb`: Individual notebooks for VGG, ResNet, Inception, and DenseNet training/evaluation.
- `best_*_solar.keras`: Fine-tuned model weights for each architecture.
- `Solar Panel Videos/`: Directory containing source and processed transition videos.

## Getting Started

### Prerequisites
- Python 3.10+
- TensorFlow 2.15+ (with GPU support recommended)
- OpenCV, NumPy, Matplotlib

### Installation
```bash
# Clone the repository (if applicable)
# Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### Usage
Run the analysis script to process a solar panel video:
```bash
python3 solar_panel_video_analysis.py
```
*Note: You can toggle models and slowdown factors within the `if __name__ == "__main__":` block in the script.*


---
*Developed for Neural Networks Case Study on Solar Panels.*
