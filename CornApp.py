import os
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 页面配置 ==========
st.set_page_config(
    page_title="玉米年份预测系统",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 2. 注入“现代活力”风格 CSS ==========
def local_css():
    st.markdown("""
        <style>
        /* 导入中文字体 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans SC', sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%); /* 渐变背景 */
        }

        /* 主标题：彩色渐变 */
        .main-title {
            background: linear-gradient(to right, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 42px;
            text-align: center;
            margin-bottom: 5px;
        }
        .sub-title {
            color: #64748b;
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
        }

        /* 现代卡片：带顶部彩条 */
        .vibrant-card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            margin-bottom: 25px;
            border-top: 5px solid #3b82f6; /* 顶部蓝色装饰条 */
        }

        /* 预测结果专用卡片颜色分支 */
        .res-card-green { border-left: 10px solid #10b981; background-color: #f0fdf4; }
        .res-card-blue { border-left: 10px solid #3b82f6; background-color: #eff6ff; }
        .res-card-orange { border-left: 10px solid #f59e0b; background-color: #fffbeb; }
        .res-card-red { border-left: 10px solid #ef4444; background-color: #fef2f2; }

        /* 按钮：深蓝色活力按钮 */
        .stButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            border-radius: 12px;
            padding: 0.6rem 2rem;
            border: none;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
            color: white;
        }

        /* 侧边栏样式 */
        section[data-testid="stSidebar"] {
            background-color: #1e293b;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ========== 3. 核心配置与资源加载 ==========
IMPUTER_SCALER_PATH = 'corn_treat.pkl'       
LNN_MODEL_PATH = 'LNNclassification.pt'     
LOG_FILE_PATH = 'user_agreement_log.txt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load(IMPUTER_SCALER_PATH)
        model = torch.jit.load(LNN_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return scaler, model
    except Exception as e:
        st.error(f"资源加载失败: {e}")
        return None, None

# ========== 4. 免责声明逻辑 ==========
def check_disclaimer():
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown('<div class="vibrant-card" style="border-top-color: #ef4444; margin-top: 50px;">', unsafe_allow_html=True)
            st.markdown("### ⚖️ 合规确认与免责声明")
            st.write("""
            本内容由人工智能模型生成，所呈现的预测结果仅为基于已有数据与算法的推演。
            
            1. **非保证性**：预测结果不构成任何形式的保证或专业建议。
            2. **误差说明**：模型可能存在计算偏差，实际结果可能与预测存在差异。
            3. **责任自担**：用户应独立做出决策，并自行承担因使用本系统引发的风险。
            """)
            if st.button("我已阅读并同意"):
                try:
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{t}] 用户同意声明\n")
                except: pass
                st.session_state.agreed = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

# ========== 5. 数据处理类 ==========
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

# ========== 6. 主程序界面 ==========
check_disclaimer()

st.markdown('<p class="main-title">玉米储存年份预测系统</p >', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于液态神经网络 (LNN) 的快速无损分析方案</p >', unsafe_allow_html=True)

scaler, model = load_resources()

if scaler and model:
    # 侧边栏：操作说明
    with st.sidebar:
        st.title("⚙️ 系统管理")
        st.info("上传含有气质风味组学数据的 Excel 或 CSV 文件进行分析。")
        if st.checkbox("显示系统日志"):
            if os.path.exists(LOG_FILE_PATH):
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    st.download_button("📥 下载审计日志", f.read(), file_name="log.txt")

    # 主体布局
    l, r = st.columns([1, 2])
    
    with l:
        st.markdown('<div class="vibrant-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 数据上传")
        uploaded_file = st.file_uploader("选择文件", type=["xlsx", "csv"], label_visibility="collapsed")
        
        if uploaded_file:
            st.success(f"已加载: {uploaded_file.name}")
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.write("数据行数:", len(df))
            
            if st.button("🚀 执行模型推理"):
                st.session_state.do_predict = True
        st.markdown('</div>', unsafe_allow_html=True)

    with r:
        if uploaded_file and 'do_predict' in st.session_state:
            with st.spinner("🧠 神经网络计算中..."):
                processor = Data_prepossing(SEQ_LENGTH=1)
                input_tensor, names = processor.prediction_pretreatment(df, scaler)

                if input_tensor is not None:
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        confs = torch.max(probs, dim=1)[0]

                    st.markdown("#### 🔍 预测报告")
                    labels = ['≤1年 (新粮)', '1-2年', '2-3年', '3年及以上 (陈粮)']
                    # 对应的颜色卡片样式
                    card_styles = ['res-card-green', 'res-card-blue', 'res-card-orange', 'res-card-red']

                    for name, p_idx, conf in zip(names, preds, confs):
                        idx = int(p_idx)
                        style = card_styles[idx]
                        
                        # 展示结果卡片
                        st.markdown(f"""
                            <div class="vibrant-card {style}" style="padding: 15px 25px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <b style="color: #64748b; font-size: 13px;">样本名称</b>
                                        <div style="font-size: 18px; font-weight: 700;">{name}</div>
                                    </div>
                                    <div style="text-align: center;">
                                        <b style="color: #64748b; font-size: 13px;">预测结果</b>
                                        <div style="font-size: 20px; font-weight: 700; color: #1e3a8a;">{labels[idx]}</div>
                                    </div>
                                    <div style="text-align: right;">
                                        <b style="color: #64748b; font-size: 13px;">置信度</b>
                                        <div style="font-size: 20px; font-weight: 700;">{conf:.1%}</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            # 默认提示
            st.info("👈 请在左侧上传数据并点击按钮开始预测。")

