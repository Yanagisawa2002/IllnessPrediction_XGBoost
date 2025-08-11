# 🏥 Intelligent Disease Prediction System

An intelligent disease prediction system based on XGBoost machine learning algorithm and Gradio web interface, capable of predicting potential diseases based on user-selected symptoms.

## 📋 Project Overview

This project is a complete medical diagnostic assistance system that predicts potential diseases by analyzing patient symptoms. The system uses XGBoost algorithm to train the model and provides a user-friendly web interface for interactive use.

## ✨ Key Features

- 🤖 **Intelligent Prediction**: Based on XGBoost machine learning algorithm
- 🌐 **Web Interface**: Modern web interface built with Gradio
- 📊 **Multi-result Display**: Shows predicted diseases, confidence levels, and probability rankings
- ⚡ **GPU Acceleration**: Supports NVIDIA GPU acceleration for training
- 🎯 **High Accuracy**: Achieves 100% accuracy on test data

## 📁 Project Structure

```
IllnessPrediction/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependency list
├── dataset.csv                  # Disease symptom dataset
├── gradio_disease_predictor.py  # Gradio web interface main program
├── xgboost_full_auto.py        # Complete XGBoost model implementation
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Web Interface

```bash
python gradio_disease_predictor.py
```

After startup, access `http://127.0.0.1:7860` in your browser to use the interface.

### Run Complete Demo

```bash
python xgboost_full_auto.py
```

## 📊 Dataset Information

- **Sample Count**: 4,920 cases
- **Symptom Features**: 131 different symptoms
- **Disease Categories**: 41 disease types
- **Data Format**: CSV format, containing disease names and corresponding symptoms

## 🎯 Usage Instructions

### Web Interface Usage

1. Launch the Gradio interface
2. Check the patient's symptoms in the symptom selection area
3. Click the "🔍 Start Prediction" button
4. View prediction results:
   - Summary of selected symptoms
   - Most likely disease with confidence level
   - Probability ranking of top 3 possible diseases

### Command Line Usage

Running `xgboost_full_auto.py` will show:
- Data loading and preprocessing process
- CPU vs GPU training performance comparison
- Model evaluation results
- Feature importance analysis
- Batch prediction demonstration

## 🔧 Technology Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Web Interface**: Gradio
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **GPU Acceleration**: CUDA (optional)

## 📈 Model Performance

- **Accuracy**: 100% (test set)
- **Prediction Speed**: 0.03ms/sample
- **Training Time**: ~1-3 seconds (depending on hardware)
- **GPU Support**: Automatic detection and GPU acceleration enablement

## 🎨 Interface Preview

The system provides a modern web interface including:
- Multi-column symptom selection area
- Real-time prediction result display
- Disease probability ranking
- One-click clear function

## 📝 Important Notes

⚠️ **Disclaimer**: This system is for learning and research purposes only and cannot replace professional medical diagnosis. If you have health concerns, please consult a professional doctor.
