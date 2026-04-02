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
    page_title="玉米储存年份预测分析系统",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 2. 注入“现代活力” UI 样式表 ==========
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans SC', -apple-system, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        /* 主标题样式 */
        .main-title {
            background: linear-gradient(to right, #1e3a8a, #2563eb, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 48px;
            text-align: center;
            margin-bottom: 0px;
            padding-top: 20px;
            letter-spacing: -1px;
        }

        /* 标签样式 */
        .interface-tag {
            background-color: #3b82f6;
            color: white;
            padding: 2px 14px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            margin-right: 12px;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        }

        .sub-title {
            color: #64748b;
            text-align: center;
            font-size: 20px;
            margin-top: 15px;
            margin-bottom: 45px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* 现代卡片 */
        .vibrant-card {
            background-color: #ffffff;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
            border: 1px solid #e2e8f0;
        }

        /* 结果卡片 */
        .res-card-0 { border-left: 8px solid #10b981; background-color: #f0fdf4; }
        .res-card-1 { border-left: 8px solid #3b82f6; background-color: #eff6ff; }
        .res-card-2 { border-left: 8px solid #f59e0b; background-color: #fffbeb; }
        .res-card-3 { border-left: 8px solid #ef4444; background-color: #fef2f2; }

        /* 按钮样式 */
        .stButton>button {
            background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            border-radius: 12px;
            padding: 0.75rem 2.5rem;
            border: none;
            font-weight: 600;
            font-size: 18px;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
            color: white;
        }

        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

local_css()

# ========== 3. 路径配置与资源加载 ==========
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

# ========== 4. 免责声明逻辑 (已修复标题显示) ==========
def check_disclaimer():
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        # 在免责界面也显示完整的系统名称
        st.markdown('<p class="main-title">🌽 玉米储存年份预测分析系统</p >', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">系统准入与合规确认</p >', unsafe_allow_html=True)
        
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown('<div class="vibrant-card" style="border-top: 6px solid #ef4444;">', unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #1e293b;'>使用协议与免责声明</h3>", unsafe_allow_html=True)
            st.write("""
            本内容由人工智能模型生成，所呈现的预测、分析或结论仅为基于已有数据与算法的推演结果，不构成任何形式的保证、承诺或专业建议。AI模型可能存在误差、偏差或对未知因素的考虑不足，实际结果可能与预测存在显著差异。
            
            **核心条款：**
            1. 用户应结合自身判断、专业咨询及实时信息独立做出决策。
            2. 用户需自行承担因使用本预测内容所引发的一切风险与责任。
            3. 开发者及提供方不对任何直接或间接损失承担责任。
            
            请谨慎使用预测结果，切勿将其作为唯一决策依据。
            """)
            if st.button("我已阅读并同意上述协议"):
                try:
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{t}] 用户同意协议并进入分析系统\n")
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
            missing = set(expected_cols) - set(df_uploaded.columns)
            if missing:
                st.error(f"数据缺少必要特征列: {missing}")
                return None, None
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

# 主界面标题
st.markdown('<p class="main-title">🌽 玉米储存年份预测分析系统</p >', unsafe_allow_html=True)
st.markdown("""
    <p class="sub-title">
        <span class="interface-tag">主界面</span> 
        基于液态神经网络 (LNN) 的自动化分析平台
    </p >
""", unsafe_allow_html=True)

scaler, model = load_resources()

if scaler and model:
    with st.sidebar:
        st.markdown("<h3 style='color:white;'>系统管理</h3>", unsafe_allow_html=True)
        if st.checkbox("查看审计日志"):
            if os.path.exists(LOG_FILE_PATH):
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    st.download_button("导出日志", f.read(), file_name="log.txt")

    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown('<div class="vibrant-card">', unsafe_allow_html=True)
        st.markdown("#### 📂 样本载入")
        file = st.file_uploader("上传 Excel/CSV", type=["xlsx", "csv"], label_visibility="collapsed")
        if file:
            df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
            st.info(f"已加载样本数: {len(df)}")
            with st.expander("数据预览"):
                st.dataframe(df.head(5))
            if st.button("🚀 执行分析预测"):
                st.session_state.run = True
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if file and 'run' in st.session_state:
            with st.spinner("系统正在计算中..."):
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
                    styles = ['res-card-0', 'res-card-1', 'res-card-2', 'res-card-3']
                    for n, p, c in zip(names, preds, confs):
                        idx = int(p)
                        st.markdown(f"""
                            <div class="vibrant-card {styles[idx]}" style="padding: 15px 25px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="flex:1;"><b>样品名称</b><br>{n}</div>
                                    <div style="flex:1; text-align: center;"><b>预测年份</b><br><span style="color:#1e3a8a;font-weight:700;">{labels[idx]}</span></div>
                                    <div style="flex:1; text-align: right;"><b>置信度</b><br>{c:.1%}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 100px; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 20px;">请在左侧上传数据并启动预测</div>', unsafe_allow_html=True)
