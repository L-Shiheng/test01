import os
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 页面配置 (必须在第一行) ==========
st.set_page_config(
    page_title="Corn Prediction System",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 2. 注入苹果官网风格 CSS ==========
def local_css():
    st.markdown("""
        <style>
        /* 全局字体设置，模拟 SF Pro */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f5f5f7; /* 苹果官网经典底色 */
        }

        /* 标题样式：大、黑、重 */
        .main-title {
            font-weight: 600;
            font-size: 48px;
            color: #1d1d1f;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: -0.02em;
        }
        .sub-title {
            font-size: 24px;
            color: #86868b;
            text-align: center;
            margin-bottom: 50px;
        }

        /* 卡片样式：极简圆角与微阴影 */
        .apple-card {
            background-color: #ffffff;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            margin-bottom: 20px;
            border: 1px solid #e5e5e7;
        }

        /* 按钮优化：蓝色胶囊状 */
        .stButton>button {
            background-color: #0071e3;
            color: white;
            border-radius: 980px; /* 彻底圆角 */
            padding: 12px 24px;
            border: none;
            font-weight: 400;
            transition: all 0.3s ease;
            width: auto;
            margin: 0 auto;
            display: block;
        }
        .stButton>button:hover {
            background-color: #0077ed;
            box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
            border: none;
            color: white;
        }

        /* 指标卡片优化 */
        [data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        /* 隐藏Streamlit页眉页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

local_css()

# ========== 3. 路径与设备配置 ==========
IMPUTER_SCALER_PATH = 'corn_treat.pkl'       
LNN_MODEL_PATH = 'LNNclassification.pt'     
LOG_FILE_PATH = 'user_agreement_log.txt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 4. 免责声明逻辑 (改为卡片式居中展示)
# ==========================================
def check_disclaimer():
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown('<p class="main-title">Legal</p >', unsafe_allow_html=True)
            st.markdown("""
                <div class="apple-card">
                    <h3 style='text-align: center; color: #1d1d1f;'>免责声明</h3>
                    <p style='color: #424245; line-height: 1.6; font-size: 15px;'>
                    本内容由人工智能模型生成，所呈现的预测、分析或结论仅为基于已有数据与算法的推演结果，不构成任何形式的保证、承诺或专业建议。AI模型可能存在误差、偏差或对未知因素的考虑不足，实际结果可能与预测存在显著差异。用户应结合自身判断、专业咨询及实时信息独立做出决策，并自行承担因使用本预测内容所引发的一切风险与责任。开发者及提供方不对任何直接或间接损失承担责任。<br><br>
                    <strong>请谨慎使用预测结果，切勿将其作为唯一决策依据。</strong>
                    </p >
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Agree and Continue"):
                try:
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{t}] 用户同意免责声明\n")
                except: pass
                st.session_state.agreed = True
                st.rerun()
        st.stop()

# ==========================================
# 5. 模型加载与预处理 (保持原有逻辑)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load(IMPUTER_SCALER_PATH)
        model = torch.jit.load(LNN_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return scaler, model
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None, None

class Data_prepossing:
    def __init__(self, SEQ_LENGTH=1):
        self.seq_length = SEQ_LENGTH
    def prediction_pretreatment(self, df_uploaded, scaler):
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_.tolist()
            prediction_data = df_uploaded[expected_cols]
        else:
            prediction_data = df_uploaded.drop(['mass', 'year', 'Skatole', 'Vanillin'], axis=1, errors='ignore')
        col_name = df_uploaded['mass'] if 'mass' in df_uploaded.columns else df_uploaded.iloc[:, 0]
        trans_data = scaler.transform(prediction_data)
        tensor_data = torch.tensor(trans_data, dtype=torch.float32)
        feature_size = tensor_data.shape[1] // self.seq_length
        seriesed_data = tensor_data.view(-1, self.seq_length, feature_size)
        return seriesed_data.to(DEVICE), col_name

# ==========================================
# 6. 主程序界面
# ==========================================
check_disclaimer()

# 页面标题
st.markdown('<p class="main-title">Corn Intelligence</p >', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于液态神经网络的玉米储存年份预测</p >', unsafe_allow_html=True)

scaler, model = load_resources()

if scaler and model:
    # 居中布局
    _, center_col, _ = st.columns([1, 4, 1])
    
    with center_col:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file (Excel/CSV)", type=["xlsx", "csv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
                
                with st.expander("View Uploaded Data"):
                    st.dataframe(df.head(), use_container_width=True)

                if st.button("Start Analysis"):
                    processor = Data_prepossing(SEQ_LENGTH=1)
                    input_tensor, names = processor.prediction_pretreatment(df, scaler)

                    if input_tensor is not None:
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            preds = torch.argmax(probs, dim=1)
                            confs = torch.max(probs, dim=1)[0]

                        st.markdown('<p style="font-size: 28px; font-weight: 600; margin-top: 40px;">Analysis Report</p >', unsafe_allow_html=True)
                        labels = ['≤1 year', '1-2 year', '2-3 year', '3+ year']

                        for name, p_idx, conf in zip(names, preds, confs):
                            # 每个结果一个精致小卡片
                            st.markdown(f"""
                                <div class="apple-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <p style="color: #86868b; font-size: 14px; margin-bottom: 5px;">Sample Name</p >
                                            <p style="font-size: 20px; font-weight: 600;">{name}</p >
                                        </div>
                                        <div style="text-align: right;">
                                            <p style="color: #86868b; font-size: 14px; margin-bottom: 5px;">Prediction</p >
                                            <p style="font-size: 20px; font-weight: 600; color: #0071e3;">{labels[int(p_idx)]}</p >
                                        </div>
                                        <div style="text-align: right;">
                                            <p style="color: #86868b; font-size: 14px; margin-bottom: 5px;">Confidence</p >
                                            <p style="font-size: 20px; font-weight: 600;">{conf:.1%}</p >
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

# 管理员日志下载 (侧边栏)
with st.sidebar:
    st.markdown("### Settings")
    if st.checkbox("Show Logs"):
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                st.download_button("Download Logs", f.read(), file_name="logs.txt")
