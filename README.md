# Navarasa Emotion Recognition System

A real-time facial emotion recognition system that detects the 9 Navarasa (classical Indian emotions) from facial expressions.

## 📋 Overview

This project combines:
- **Facial Detection**: OpenCV Haar Cascade for real-time face detection
- **Emotion Recognition**: TensorFlow/Keras deep learning model trained on 9 emotions
- **Multi-source Input**: Support for local webcam or IP camera streams

### Navarasa Emotions

The system recognizes the 9 classical Navarasa emotions:

| Emotion | Meaning |
|---------|----------|
| ADBHUTA | Wonder/Surprise |
| BHAYANAKA | Fear |
| BIBHATSYA | Disgust |
| HASYA | Joy/Laughter |
| KARUNA | Sorrow/Compassion |
| RAUDRA | Anger |
| SHANTA | Peace/Tranquility |
| SHRINGARA | Love/Romance |
| VEERA | Courage/Heroism |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam or IP camera

### Installation

1. **Clone/Download the repository**
   ```bash
   cd "Navarasa-Detection"
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Basic Emotion Recognition (Display Only)

```bash
# Use default webcam (device 0)
python detect.py

# Use specific camera device
python detect.py 0

# Use IP camera stream
python detect.py http://192.168.1.100:8080/video
```

#### Controls

- **'q'** - Quit the application
- **'r'** - Unlock detection (reset 3-second lock)

## 📁 Project Structure

```
Navarasa 1/
├── detect.py                              # Main emotion recognition script
├── navarasa_emotion_model_split1.h5       # Pre-trained Keras model
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
├── extracted_dataset/                     # Training dataset
│   └── NAVRASA FACIAL EMOTION IMAGE DATA/
│       ├── training/                      # Training images (9 emotions)
│       └── testing/                       # Test images (9 emotions)
└── requirements.txt                       # Python dependencies
```

## 🔧 Configuration

### Model Selection

## 📊 Features

- ✅ Real-time facial emotion detection
- ✅ Support for local and IP camera streams
- ✅ Multi-face detection capability
- ✅ 3-second emotion lock to prevent rapid switching
- ✅ Preprocessing: grayscale conversion, resizing (48x48), normalization
- ✅ Haarcascade face detection with high accuracy

## 🔗 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **OpenCV** (`opencv-python`) - Computer vision and face detection
- **TensorFlow** (`tensorflow`) - Deep learning framework for emotion model
- **NumPy** (`numpy`) - Numerical computing
- **PyTorch** (`torch`) - Alternative model support

## 📈 Model Details

### Input
- Image size: 48×48 pixels (grayscale)
- Normalization: 0-1 range

### Output
- 9 emotion classes (Navarasa emotions)
- Softmax probability distribution

## ⚙️ System Requirements

### Minimum
- CPU: Intel i5 / Ryzen 5
- RAM: 4 GB
- Webcam or IP camera

### Recommended
- CPU: Intel i7 / Ryzen 7
- GPU: NVIDIA CUDA-capable (for faster inference)
- RAM: 8+ GB

## 🐛 Troubleshooting

### "Cannot open camera"
- Check camera is connected and not in use by another application
- For IP camera: verify URL is correct and both devices are on same network

### "AttributeError: module 'cv2' has no attribute 'CascadeClassifier'"
- Reinstall OpenCV: `pip install --upgrade opencv-python`

### GPU not detected
- Install CUDA Toolkit and cuDNN for GPU acceleration
- For TensorFlow GPU: `pip install tensorflow[and-cuda]`

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributor information]

## 🔮 Future Enhancements

- [ ] Multi-face tracking with individual emotion labels
- [ ] Emotion history/timeline visualization
- [ ] Audio emotion integration
- [ ] Real-time model training on user data
- [ ] Web interface for monitoring
- [ ] Mobile app for remote monitoring
- [ ] Emotion confidence score display

## 📞 Support

For issues or questions, please check:
1. The troubleshooting section above
2. OpenCV documentation: https://docs.opencv.org/
3. TensorFlow documentation: https://www.tensorflow.org/

---

**Created**: 2024-2026  
**Last Updated**: May 2026
