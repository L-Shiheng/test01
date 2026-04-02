import os
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 路径与设备配置 ==========
# 如果部署到 GitHub，建议改为相对路径，例如：IMPUTER_SCALER_PATH = 'corn_treat.pkl'
IMPUTER_SCALER_PATH = r'D:\AIOnline\corn\corn_treat.pkl'       
LNN_MODEL_PATH = r'D:\AIOnline\corn\LNNclassification.pt'     
LOG_FILE_PATH = r'D:\AIOnline\corn\user_agreement_log.txt'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 免责声明与合规记录逻辑
# ==========================================
def check_disclaimer():
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        st.title("🌽 玉米储存年份预测系统")
        st.warning("### ⚖️ 免责声明")
        st.info("""
        本内容由人工智能模型生成，所呈现的预测、分析或结论仅为基于已有数据与算法的推演结果，不构成任何形式的保证、承诺或专业建议。AI模型可能存在误差、偏差或对未知因素的考虑不足，实际结果可能与预测存在显著差异。用户应结合自身判断、专业咨询及实时信息独立做出决策，并自行承担因使用本预测内容所引发的一切风险与责任。开发者及提供方不对任何直接或间接损失承担责任。

        **请谨慎使用预测结果，切勿将其作为唯一决策依据。**
        """)
        
        if st.button("我已阅读并同意以上声明", type="primary"):
            # 记录日志
            try:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{t}] 用户同意了免责声明并进入系统\n")
            except:
                pass # 防止权限问题导致崩溃
                
            st.session_state.agreed = True
            st.rerun()
        st.stop()

# ==========================================
# 3. 数据预处理类 (优化版)
# ==========================================
class Data_prepossing:
    def __init__(self, SEQ_LENGTH=1):
        self.seq_length = SEQ_LENGTH
        
    def prediction_pretreatment(self, df_uploaded: pd.DataFrame, scaler):
        # --- 核心修复：自动对齐特征列 ---
        # 1. 获取预处理器需要的特征名单
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_.tolist()
            # 检查上传的列是否完整
            missing = set(expected_cols) - set(df_uploaded.columns)
            if missing:
                st.error(f"❌ 上传的文件缺少必要特征列: {missing}")
                return None, None
            # 自动提取并排序，忽略多余的 'year', 'Skatole' 等列
            prediction_data = df_uploaded[expected_cols]
        else:
            # 备选：如果scaler没存名字，则剔除已知的非特征列
            prediction_data = df_uploaded.drop(['mass', 'year', 'Skatole', 'Vanillin'], axis=1, errors='ignore')

        # 2. 获取样品名称（假设第一列是名称，或者有 'mass' 列）
        col_name = df_uploaded['mass'] if 'mass' in df_uploaded.columns else df_uploaded.iloc[:, 0]

        # 3. 标准化转换
        trans_data = scaler.transform(prediction_data)

        # 4. 转换为 Tensor 并调整维度 [Batch, Seq, Features]
        tensor_data = torch.tensor(trans_data, dtype=torch.float32)
        # 重新整理形状：(样本数, 序列长度, 每个序列的特征数)
        # 这里根据你之前的逻辑，SEQ_LENGTH=1
        feature_size = tensor_data.shape[1] // self.seq_length
        seriesed_data = tensor_data.view(-1, self.seq_length, feature_size)
        
        return seriesed_data.to(DEVICE), col_name

# ==========================================
# 4. 模型加载 (带缓存)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # 加载预处理器
        scaler = joblib.load(IMPUTER_SCALER_PATH)
        # 加载 LNN 模型 (TorchScript)
        model = torch.jit.load(LNN_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return scaler, model
    except Exception as e:
        st.error(f"资源加载失败: {e}")
        return None, None

# ==========================================
# 5. 主程序界面
# ==========================================
# 权限检查
check_disclaimer()

st.title("🌽 基于液态神经网络的玉米储存年份预测")

scaler, model = load_resources()

if scaler and model:
    uploaded_file = st.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"])

    if uploaded_file:
        # 读取数据
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.write("### 数据预览")
            st.dataframe(df.head())

            if st.button("开始分析预测", type="primary"):
                processor = Data_prepossing(SEQ_LENGTH=1)
                input_tensor, names = processor.prediction_pretreatment(df, scaler)

                if input_tensor is not None:
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        confs = torch.max(probs, dim=1)[0]

                    # 结果展示
                    st.divider()
                    st.subheader("🔮 预测报告")
                    labels = ['≤1 year', '1-2 year', '2-3 year', '3+ year']

                    for name, p_idx, conf in zip(names, preds, confs):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("样品", str(name))
                        c2.metric("预测年份", labels[int(p_idx)])
                        c3.metric("置信度", f"{conf:.1%}")
                        
                        if conf > 0.8:
                            st.success("预测结果可靠")
                        elif conf > 0.5:
                            st.warning("建议结合实验复核")
                        else:
                            st.error("结果仅供参考")
                        st.write("---")
        except Exception as e:
            st.error(f"运行时出错: {e}")
