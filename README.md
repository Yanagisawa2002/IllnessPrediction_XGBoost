## 📊 Dataset Information

- **Samples**: 4,920 medical cases
- **Symptom Features**: 131 different symptoms
- **Disease Classes**: 41 disease types
- **Data Format**: CSV containing disease names and corresponding symptoms

## 🎯 Usage Guide

### Web Interface

1. Launch Gradio interface
2. Select patient symptoms in the symptom selection area
3. Click the "🔍 Start Prediction" button
4. View results:
   - Summary of selected symptoms
   - Most likely disease with confidence level
   - Top 3 probable diseases with probability ranking

### Command Line Usage

Run `xgboost_full_auto.py` to see:
- Data loading and preprocessing
- CPU vs GPU training performance comparison
- Model evaluation results
- Feature importance analysis
- Batch prediction demo

## 🔧 Technology Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Web Interface**: Gradio
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **GPU Acceleration**: CUDA (optional)

## 📈 Model Performance

- **Accuracy**: 100% (test set)
- **Prediction Speed**: 0.03ms/sample
- **Training Time**: ~1-3 seconds (hardware-dependent)
- **GPU Support**: Automatically detects and enables GPU acceleration

## 🎨 Interface Preview

The system provides a modern web interface featuring:
- Multi-column symptom selection area
- Real-time prediction results
- Disease probability ranking
- One-click reset functionality

## 📝 Important Notes

⚠️ **Disclaimer**: This system is for educational and research purposes only and is not a substitute for professional medical diagnosis. Consult qualified healthcare providers for health concerns.
