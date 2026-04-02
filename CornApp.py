import os
import joblib
import torch
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 定义设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 导入路径配置 ==========
IMPUTER_SCALER_PATH = r'D:\AIOnline\corn\corn_treat.pkl'       # 预处理器（标准化/插补模型）
LNN_MODEL_PATH = r'D:\AIOnline\corn\LNNclassification.pt'     # 训练好的LSTM模型权重

# ==========================================
# 1. 模型预处理类定义
# ==========================================
# 读取数据并进行预处理，生成训练集和测试集的tensor数据
class Data_prepossing:
    def __init__(self,SEQ_LENGTH:int=1,SEQ_SIZE:int=80):
        super(Data_prepossing, self).__init__()
        self.seq_length = SEQ_LENGTH
        self.seq_size = SEQ_SIZE
        
    def load_imputer(self):
        # 从配置读取预处理器路径
        scaler_path = IMPUTER_SCALER_PATH
        if not os.path.exists(scaler_path):
            st.error(f"❌ 找不到预处理文件：{scaler_path}")
            return None
        try:
            # 使用 joblib 加载 sklearn 的对象
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            st.error(f"预处理器加载出错: {e}")
            return None
        
    def create_tensors(self,X):
        # 生成tensor数据
        X_tensor = torch.tensor(X,dtype=torch.float32)

        return X_tensor

    # 创建序列数据
    def create_series(self, X, SEQ_LENGTH=1):
        # 创建序列样本长度
        SEQ_SIZE = int(X.shape[1]/SEQ_LENGTH)
        # 创建训练集Dataloader
        dataset = X.view(-1,SEQ_LENGTH,SEQ_SIZE)

        return dataset

    # 数据对齐和整合
    def prediction_pretreatment(self,uploaded_file: pd.DataFrame):
        # 删去辅助数据
        prediction = uploaded_file.drop(['mass', 'year'], axis=1).iloc[:,:-2].values

        # 获取样品名称
        col_name = uploaded_file['mass']

        scaler = self.load_imputer()
        if scaler is None:
            st.error("❌ 预处理器未正确加载，无法进行数据转换。")
            return None, None
        
        trans_prediction = scaler.transform(prediction)

        # 转换为tensor数据
        prediction_tensor = self.create_tensors(trans_prediction)

        # 转换为seriesed_tensor 数据
        prediction_seriesed = self.create_series(prediction_tensor, SEQ_LENGTH=self.seq_length).to(DEVICE)
        
        return prediction_seriesed, col_name

# ==========================================
# 2. streamlit界面及模型上载
# ==========================================
@st.cache_resource
def load_deep_model():
    model_path = LNN_MODEL_PATH  # 从配置读取
    if not os.path.exists(model_path):
        st.error(f"❌ 找不到 LNN 模型文件：{model_path}")
        return None
    try:
        # 加载 TorchScript 模型
        model = torch.jit.load(LNN_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"LNN 模型加载出错: {e}")
        return None

# Streamlit界面
st.title("基于气质风味组学数据库和液态神经网络的玉米储存年份预测系统")
uploaded_file = st.file_uploader("上传一个或者多个样品的excel文件", type=["xlsx", "csv"], accept_multiple_files=False)

if uploaded_file:
    if st.button("开始训练并预测"):
        # 数据预处理
        prepossessor = Data_prepossing(SEQ_LENGTH=1,SEQ_SIZE=80)
        df_uploaded = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        input_tensor, col_name = prepossessor.prediction_pretreatment(df_uploaded)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化加载
        model = load_deep_model()
        if model is None:
            st.error("❌ 模型加载失败，无法进行预测。")
        else:
            model = model.to(DEVICE)

        # 预测
        predicted_class = None
        confidence = None
        # 进行预测并计算置信度
        if model is not None and input_tensor is not None:
            with torch.no_grad():
                output = model(input_tensor.to(DEVICE))
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0].cpu()

        elif input_tensor is None:
            st.error("❌ 输入数据预处理失败，无法进行预测。")
            
        else:
            st.error("❌ 模型未正确加载，无法进行预测。")

        # ==========================================
        # 3. 结果展示
        # ==========================================
        st.divider()
        st.subheader("🔮 分析报告")
        group_name = ['≤1 year', '1-2 year', '2-3 year', '3+ year']  # 类别名称列表

        if col_name is not None and predicted_class is not None and confidence is not None:
            for col, m, n in zip(col_name, predicted_class, confidence):
                # 转换类型
                class_idx = int(m.item())     # 获取整数类别索引并确保为int
                conf_val = n.item()           # 获取该样本的置信度（浮点数）

                col1, col2, col3 = st.columns(3)
                col1.metric("样品名称", f"{col}")
                col2.metric("预测类别", f"{group_name[class_idx]}")
                col3.metric("置信度", f"{conf_val:.1%}")

                # 针对当前样本的信心评估
                if conf_val > 0.8:
                    st.success("✅ 模型对此结果非常有信心。")
                elif conf_val > 0.5:
                    st.warning("⚠️ 模型信心一般，建议结合其它方法判断。")
                else:
                    st.error("❌ 模型信心不足，结果仅供参考。")
        else:
            st.error("❌ 样品名称数据缺失或预测结果未生成，无法展示结果。")
    else:
        st.warning("⚠️ 请上传数据文件，并确保文件存在。")
