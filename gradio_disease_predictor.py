import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gradio as gr
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class GradioDiseasePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.symptoms_list = []
        self.model = None
        self.label_encoder = LabelEncoder()
        self.X = None
        self.y = None
        
    def load_and_prepare_data(self):
        """加载数据并准备症状列表"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        
        # 提取所有症状
        all_symptoms = set()
        symptom_columns = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        for col in symptom_columns:
            symptoms_in_col = self.df[col].dropna().str.strip()
            # 过滤掉空字符串
            symptoms_in_col = symptoms_in_col[symptoms_in_col != '']
            all_symptoms.update(symptoms_in_col.unique())
        
        # 移除空值和空字符串
        self.symptoms_list = sorted([s for s in all_symptoms if s and s.strip()])
        print(f"发现 {len(self.symptoms_list)} 种不同症状")
        
        # 准备训练数据
        self.prepare_training_data()
        
    def prepare_training_data(self):
        """准备训练数据"""
        print("正在准备训练数据...")
        
        # 创建症状特征矩阵
        symptom_data = []
        diseases = []
        
        for _, row in self.df.iterrows():
            # 获取该行的所有症状
            patient_symptoms = []
            for col in [col for col in self.df.columns if col.startswith('Symptom_')]:
                if pd.notna(row[col]) and row[col].strip():
                    patient_symptoms.append(row[col].strip())
            
            # 创建二进制特征向量
            feature_vector = [1 if symptom in patient_symptoms else 0 for symptom in self.symptoms_list]
            symptom_data.append(feature_vector)
            diseases.append(row['Disease'])
        
        self.X = np.array(symptom_data)
        self.y = self.label_encoder.fit_transform(diseases)
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"疾病类别数: {len(self.label_encoder.classes_)}")
        
    def load_model(self, model_path='trained_model.pkl'):
        """加载已保存的模型"""
        if os.path.exists(model_path):
            print(f"正在加载已保存的模型: {model_path}...")
            
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # 从保存的数据中提取组件
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                self.symptoms_list = model_data['symptoms_list']
                
                # 显示模型信息
                model_info = model_data['model_info']
                print(f"✅ 模型加载完成！")
                print(f"📊 模型信息: {model_info['n_samples']} 样本, {model_info['n_features']} 特征, {model_info['n_diseases']} 疾病")
                print(f"🎯 模型准确率: {model_info['accuracy']:.4f}")
                print(f"🕐 训练时间: {model_info['training_time']}")
                
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        else:
            print(f"⚠️ 未找到模型文件: {model_path}，将重新训练...")
            return False
    
    def train_model(self):
        """训练XGBoost模型"""
        print("正在训练XGBoost模型...")
        
        # 检查GPU可用性
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            tree_method = 'gpu_hist' if result.returncode == 0 else 'hist'
            device = 'cuda' if result.returncode == 0 else 'cpu'
        except:
            tree_method = 'hist'
            device = 'cpu'
        
        # 训练模型
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method=tree_method,
            device=device,
            random_state=42
        )
        
        self.model.fit(self.X, self.y)
        print("模型训练完成！")
        
    def predict_disease(self, *selected_symptoms):
        """根据选择的症状预测疾病"""
        if not any(selected_symptoms):
            return "请至少选择一个症状", "", ""
        
        # 获取选中的症状
        patient_symptoms = [self.symptoms_list[i] for i, selected in enumerate(selected_symptoms) if selected]
        
        if not patient_symptoms:
            return "请至少选择一个症状", "", ""
        
        # 创建特征向量
        feature_vector = np.array([[1 if symptom in patient_symptoms else 0 for symptom in self.symptoms_list]])
        
        # 预测
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # 获取预测的疾病名称
        predicted_disease = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        # 获取前3个最可能的疾病
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = []
        for idx in top_3_indices:
            disease_name = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            top_3_diseases.append(f"{disease_name}: {prob:.2f}%")
        
        # 格式化输出
        selected_symptoms_text = "选择的症状: " + ", ".join(patient_symptoms)
        prediction_text = f"预测疾病: {predicted_disease}\n置信度: {confidence:.2f}%"
        top_3_text = "前3种可能疾病:\n" + "\n".join(top_3_diseases)
        
        return selected_symptoms_text, prediction_text, top_3_text
    
    def create_interface(self):
        """创建Gradio界面"""
        print("正在创建Gradio界面...")
        
        # 创建症状复选框
        symptom_checkboxes = []
        for symptom in self.symptoms_list:
            checkbox = gr.Checkbox(label=symptom, value=False)
            symptom_checkboxes.append(checkbox)
        
        # 创建界面
        with gr.Blocks(title="疾病预测系统", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🏥 智能疾病预测系统")
            gr.Markdown("### 请选择您当前的症状，系统将基于XGBoost模型为您预测可能的疾病")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 症状选择")
                    gr.Markdown(f"共有 {len(self.symptoms_list)} 种症状可选择")
                    
                    # 将症状分成多列显示
                    symptoms_per_column = 25
                    num_columns = (len(self.symptoms_list) + symptoms_per_column - 1) // symptoms_per_column
                    
                    symptom_inputs = []
                    for col_idx in range(num_columns):
                        with gr.Column():
                            start_idx = col_idx * symptoms_per_column
                            end_idx = min((col_idx + 1) * symptoms_per_column, len(self.symptoms_list))
                            
                            for i in range(start_idx, end_idx):
                                checkbox = gr.Checkbox(label=self.symptoms_list[i], value=False)
                                symptom_inputs.append(checkbox)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 预测结果")
                    
                    predict_btn = gr.Button("🔍 开始预测", variant="primary", size="lg")
                    
                    selected_symptoms_output = gr.Textbox(
                        label="选择的症状",
                        lines=3,
                        interactive=False
                    )
                    
                    prediction_output = gr.Textbox(
                        label="预测结果",
                        lines=3,
                        interactive=False
                    )
                    
                    top_diseases_output = gr.Textbox(
                        label="可能疾病排名",
                        lines=5,
                        interactive=False
                    )
                    
                    clear_btn = gr.Button("🔄 清除选择", variant="secondary")
            
            # 绑定预测功能
            predict_btn.click(
                fn=self.predict_disease,
                inputs=symptom_inputs,
                outputs=[selected_symptoms_output, prediction_output, top_diseases_output]
            )
            
            # 清除功能
            def clear_all():
                return [False] * len(symptom_inputs) + ["", "", ""]
            
            clear_btn.click(
                fn=clear_all,
                outputs=symptom_inputs + [selected_symptoms_output, prediction_output, top_diseases_output]
            )
            
            gr.Markdown("### 📊 系统信息")
            n_samples = self.df.shape[0] if self.df is not None else "已加载"
            gr.Markdown(f"- 训练数据: {n_samples} 个样本")
            gr.Markdown(f"- 症状特征: {len(self.symptoms_list)} 种")
            gr.Markdown(f"- 疾病类别: {len(self.label_encoder.classes_)} 种")
            gr.Markdown("- 模型: XGBoost分类器")
        
        return interface

def main():
    # 初始化预测器
    predictor = GradioDiseasePredictor('dataset.csv')
    
    # 尝试加载已保存的模型
    if not predictor.load_model():
        # 如果没有已保存的模型，则加载数据并训练
        predictor.load_and_prepare_data()
        predictor.train_model()
    
    # 创建界面
    interface = predictor.create_interface()
    
    # 启动界面
    print("\n=== 启动Gradio界面 ===")
    print("界面将在浏览器中打开...")
    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()