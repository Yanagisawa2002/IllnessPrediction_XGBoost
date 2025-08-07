import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import time
import os
import pickle
warnings.filterwarnings('ignore')

class FullAutoXGBoostDemo:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.gpu_available = self.check_gpu_availability()
        
    def check_gpu_availability(self):
        """检查GPU是否可用"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ 检测到NVIDIA GPU，将启用GPU加速")
                return True
            else:
                print("⚠ 未检测到NVIDIA GPU，使用CPU训练")
                return False
        except:
            print("⚠ 无法检测GPU状态，使用CPU训练")
            return False
    
    def load_and_explore_data(self):
        """加载数据并进行初步探索"""
        print("=== XGBoost完全自动化演示 - 数据加载 ===")
        self.df = pd.read_csv(self.data_path)
        
        print(f"数据集形状: {self.df.shape}")
        print(f"疾病类型数量: {self.df['Disease'].nunique()}")
        
        return self.df
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")
        
        # 获取症状列
        symptom_columns = [col for col in self.df.columns if col.startswith('Symptom_')]
        print(f"症状列数量: {len(symptom_columns)}")
        
        # 收集所有非空的唯一症状
        all_symptoms = set()
        for col in symptom_columns:
            symptoms_in_col = self.df[col].dropna().unique()
            all_symptoms.update([s.strip() for s in symptoms_in_col if pd.notna(s) and s.strip() != ''])
        
        all_symptoms = sorted(list(all_symptoms))
        print(f"总共发现 {len(all_symptoms)} 种不同症状")
        
        # 创建二进制特征矩阵
        feature_matrix = pd.DataFrame(0, index=self.df.index, columns=all_symptoms)
        
        for idx, row in self.df.iterrows():
            for col in symptom_columns:
                symptom_value = row[col]
                if pd.notna(symptom_value) and str(symptom_value).strip() != '':
                    symptom = str(symptom_value).strip()
                    if symptom in all_symptoms:
                        feature_matrix.loc[idx, symptom] = 1
        
        self.X = feature_matrix
        self.y = self.label_encoder.fit_transform(self.df['Disease'])
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"疾病类别数: {len(self.label_encoder.classes_)}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print(f"\n=== 数据划分 ===")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def compare_training_methods(self):
        """对比不同训练方法的性能"""
        print("\n=== 训练方法性能对比 ===")
        
        methods = []
        
        # 方法1: CPU基础训练
        cpu_params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        print("\n🖥️  方法1: CPU基础训练")
        start_time = time.time()
        cpu_model = xgb.XGBClassifier(**cpu_params)
        cpu_model.fit(self.X_train, self.y_train)
        cpu_time = time.time() - start_time
        cpu_pred = cpu_model.predict(self.X_test)
        cpu_accuracy = accuracy_score(self.y_test, cpu_pred)
        
        methods.append({
            'name': 'CPU基础',
            'time': cpu_time,
            'accuracy': cpu_accuracy,
            'model': cpu_model
        })
        print(f"   训练时间: {cpu_time:.2f}秒, 准确率: {cpu_accuracy:.4f}")
        
        # 方法2: GPU训练（如果可用）
        if self.gpu_available:
            gpu_params = cpu_params.copy()
            gpu_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            
            print("\n🚀 方法2: GPU加速训练")
            start_time = time.time()
            gpu_model = xgb.XGBClassifier(**gpu_params)
            gpu_model.fit(self.X_train, self.y_train)
            gpu_time = time.time() - start_time
            gpu_pred = gpu_model.predict(self.X_test)
            gpu_accuracy = accuracy_score(self.y_test, gpu_pred)
            
            methods.append({
                'name': 'GPU加速',
                'time': gpu_time,
                'accuracy': gpu_accuracy,
                'model': gpu_model
            })
            print(f"   训练时间: {gpu_time:.2f}秒, 准确率: {gpu_accuracy:.4f}")
            
            # 计算加速比
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"   GPU加速倍数: {speedup:.2f}x")
        
        # 方法3: 优化参数训练
        optimized_params = cpu_params.copy()
        optimized_params.update({
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        })
        
        if self.gpu_available:
            optimized_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
        
        print("\n⚡ 方法3: 优化参数训练")
        start_time = time.time()
        opt_model = xgb.XGBClassifier(**optimized_params)
        opt_model.fit(self.X_train, self.y_train)
        opt_time = time.time() - start_time
        opt_pred = opt_model.predict(self.X_test)
        opt_accuracy = accuracy_score(self.y_test, opt_pred)
        
        methods.append({
            'name': '优化参数',
            'time': opt_time,
            'accuracy': opt_accuracy,
            'model': opt_model
        })
        print(f"   训练时间: {opt_time:.2f}秒, 准确率: {opt_accuracy:.4f}")
        
        # 选择最佳模型
        best_method = max(methods, key=lambda x: x['accuracy'])
        self.model = best_method['model']
        
        print(f"\n🏆 最佳方法: {best_method['name']}")
        print(f"   准确率: {best_method['accuracy']:.4f}")
        print(f"   训练时间: {best_method['time']:.2f}秒")
        
        return methods, best_method
    
    def train_with_detailed_progress(self):
        """带详细进度的训练演示"""
        print("\n=== 详细进度训练演示 ===")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 50,  # 减少轮数以便快速演示
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 1
        }
        
        if self.gpu_available:
            params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            print("✓ 使用GPU加速训练")
        else:
            params['tree_method'] = 'hist'
            print("使用CPU优化训练")
        
        progress_model = xgb.XGBClassifier(**params)
        
        print("\n开始训练，显示每5轮进度...")
        start_time = time.time()
        
        # 训练并显示进度
        progress_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=5  # 每5轮显示一次
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 详细进度训练完成！耗时: {training_time:.2f}秒")
        
        # 评估进度训练模型
        progress_pred = progress_model.predict(self.X_test)
        progress_accuracy = accuracy_score(self.y_test, progress_pred)
        print(f"进度训练模型准确率: {progress_accuracy:.4f}")
        
        return progress_model
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\n=== 模型性能评估 ===")
        
        # 预测性能测试
        start_time = time.time()
        y_pred = self.model.predict(self.X_test)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"预测耗时: {prediction_time:.4f}秒")
        print(f"平均每样本预测时间: {prediction_time/len(self.X_test)*1000:.2f}ms")
        
        # 批量预测性能测试
        print("\n批量预测性能测试:")
        batch_sizes = [1, 10, 100, len(self.X_test)]
        
        for batch_size in batch_sizes:
            if batch_size > len(self.X_test):
                continue
                
            test_batch = self.X_test.iloc[:batch_size]
            
            start_time = time.time()
            predictions = self.model.predict(test_batch)
            batch_time = time.time() - start_time
            
            print(f"  批量大小 {batch_size:4d}: {batch_time*1000:6.2f}ms (平均 {batch_time/batch_size*1000:6.2f}ms/样本)")
        
        return accuracy
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 最重要症状:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
        
        return feature_importance
    
    def predict_disease_batch(self, test_cases):
        """批量疾病预测演示"""
        print(f"\n=== 批量疾病预测演示 ===")
        
        for i, symptoms_list in enumerate(test_cases, 1):
            print(f"\n--- 测试案例 {i} ---")
            print(f"输入症状: {symptoms_list}")
            
            # 创建特征向量
            feature_vector = pd.DataFrame(0, index=[0], columns=self.X.columns)
            
            valid_symptoms = []
            for symptom in symptoms_list:
                if symptom in self.X.columns:
                    feature_vector.loc[0, symptom] = 1
                    valid_symptoms.append(symptom)
            
            if not valid_symptoms:
                print("❌ 没有有效症状用于预测")
                continue
            
            print(f"有效症状: {valid_symptoms}")
            
            # 预测
            start_time = time.time()
            prediction_encoded = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            prediction_time = time.time() - start_time
            
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # 获取概率最高的几个疾病
            disease_names = self.label_encoder.classes_
            prob_df = pd.DataFrame({
                'disease': disease_names,
                'probability': probabilities
            }).sort_values('probability', ascending=False)
            
            print(f"🎯 预测结果: {prediction}")
            print(f"🔥 置信度: {probabilities[prediction_encoded]:.4f}")
            print(f"⚡ 预测耗时: {prediction_time*1000:.2f}ms")
            print(f"📊 Top 3 可能疾病:")
            for j, row in prob_df.head(3).iterrows():
                print(f"   {j+1}. {row['disease']}: {row['probability']:.4f}")
    
    def performance_summary(self):
        """性能总结"""
        print("\n" + "="*60)
        print("🎉 XGBoost疾病预测系统性能总结")
        print("="*60)
        
        # 系统信息
        print(f"💻 计算环境: {'GPU加速' if self.gpu_available else 'CPU优化'}")
        print(f"📊 数据规模: {self.X.shape[0]} 样本, {self.X.shape[1]} 特征")
        print(f"🏥 疾病类别: {len(self.label_encoder.classes_)} 种")
        
        # 模型性能
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🎯 模型准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 预测速度
        start_time = time.time()
        self.model.predict(self.X_test[:100])
        speed_time = time.time() - start_time
        print(f"⚡ 预测速度: {speed_time/100*1000:.2f}ms/样本")
        
        print("\n✅ 系统已准备就绪，可用于实际疾病预测！")
    
    def save_model(self, model_path='trained_model.pkl'):
        """保存训练好的模型和相关数据"""
        print(f"\n💾 正在保存模型到 {model_path}...")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'symptoms_list': list(self.X.columns),
            'feature_names': list(self.X.columns),
            'model_info': {
                'accuracy': accuracy_score(self.y_test, self.model.predict(self.X_test)),
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'n_diseases': len(self.label_encoder.classes_),
                'diseases': list(self.label_encoder.classes_),
                'training_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 模型已成功保存到 {model_path}")
        print(f"📊 模型信息: {model_data['model_info']['n_samples']} 样本, {model_data['model_info']['n_features']} 特征, {model_data['model_info']['n_diseases']} 疾病")
        print(f"🎯 模型准确率: {model_data['model_info']['accuracy']:.4f}")

# 主演示函数
def main():
    print("🚀 XGBoost疾病预测系统 - 完全自动化演示")
    print("特性: GPU加速 + 实时进度 + 性能对比 + 批量预测")
    print("="*60)
    
    # 创建模型实例
    demo = FullAutoXGBoostDemo('dataset.csv')
    
    # 1. 数据加载与预处理
    demo.load_and_explore_data()
    demo.preprocess_data()
    demo.split_data()
    
    # 2. 训练方法性能对比
    methods, best_method = demo.compare_training_methods()
    
    # 3. 详细进度训练演示
    demo.train_with_detailed_progress()
    
    # 4. 模型评估
    demo.evaluate_model()
    
    # 5. 特征重要性分析
    demo.feature_importance_analysis()
    
    # 6. 批量预测演示
    test_cases = [
        ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        ['fever', 'cough', 'fatigue'],
        ['stomach_pain', 'nausea', 'vomiting'],
        ['chest_pain', 'shortness_of_breath'],
        ['headache', 'dizziness', 'blurred_vision'],
        ['muscle_pain', 'joint_pain', 'weakness']
    ]
    
    demo.predict_disease_batch(test_cases)
    
    # 7. 性能总结
    demo.performance_summary()
    
    # 8. 保存模型
    demo.save_model('trained_model.pkl')
    
    return demo

if __name__ == "__main__":
    model = main()