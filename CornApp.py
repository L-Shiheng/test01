import os
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 页面配置与基础设置 ==========
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
        /* 导入中文字体 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans SC', -apple-system, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        /* 主标题：苹果风格渐变 */
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

        /* [主界面] 标签样式 */
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

        /* 现代圆角卡片 */
        .vibrant-card {
            background-color: #ffffff;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
            border: 1px solid #e2e8f0;
        }

        /* 结果卡片的不同色彩边框 (基于年份) */
        .res-card-0 { border-left: 8px solid #10b981; background-color: #f0fdf4; } /* 新粮 */
        .res-card-1 { border-left: 8px solid #3b82f6; background-color: #eff6ff; } /* 1-2年 */
        .res-card-2 { border-left: 8px solid #f59e0b; background-color: #fffbeb; } /* 2-3年 */
        .res-card-3 { border-left: 8px solid #ef4444; background-color: #fef2f2; } /* 陈粮 */

        /* 苹果味的大按钮 */
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

        /* 侧边栏深色风格 */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        
        /* 隐藏Streamlit默认页眉 */
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

local_css()

# ========== 3. 路径配置与资源加载 ==========
# 云端部署建议统一使用相对路径
IMPUTER_SCALER_PATH = 'corn_treat.pkl'       
LNN_MODEL_PATH = 'LNNclassification.pt'     
LOG_FILE_PATH = 'user_agreement_log.txt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        st.error(f"⚠️ 系统核心资源加载失败，请检查文件是否存在: {e}")
        return None, None

# ========== 4. 免责声明与审计日志 ==========
def check_disclaimer():
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown('<div class="vibrant-card" style="border-top: 6px solid #ef4444; margin-top: 60px;">', unsafe_allow_html=True)
            st.markdown("### ⚖️ 合规确认与免责声明")
            st.write("""
            欢迎使用本分析系统。在开始前，请知悉以下条款：
            
            1. **算法性质**：本系统结果由液态神经网络模型生成，仅供科研与数据参考，不构成法定鉴定结论。
            2. **误差说明**：由于样本多样性，预测结果可能存在统计学误差。
            3. **责任界定**：用户依据本系统结果做出的商业或专业决策，风险由用户自行承担。
            """)
            if st.button("我已阅读并同意上述条款"):
                try:
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{t}] 访客授权同意声明并进入系统\n")
                except: pass
                st.session_state.agreed = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

# ========== 5. 高级数据预处理类 ==========
class Data_prepossing:
    def __init__(self, SEQ_LENGTH=1):
        self.seq_length = SEQ_LENGTH

    def prediction_pretreatment(self, df_uploaded, scaler):
        try:
            # 自动对齐特征列，解决 Skatole/Vanillin/year 等多余列报错问题
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = scaler.feature_names_in_.tolist()
                # 检查是否缺少列
                missing = set(expected_cols) - set(df_uploaded.columns)
                if missing:
                    st.error(f"❌ 上传数据缺少模型必需的特征列: {missing}")
                    return None, None
                # 只提取模型认识的列，并按正确顺序排列
                prediction_data = df_uploaded[expected_cols]
            else:
                # 兜底：手动排除非特征列
                prediction_data = df_uploaded.drop(['mass', 'year', 'Skatole', 'Vanillin'], axis=1, errors='ignore')

            # 提取样品名称（默认取 'mass' 列或第一列）
            col_name = df_uploaded['mass'] if 'mass' in df_uploaded.columns else df_uploaded.iloc[:, 0]

            # 预处理转换
            trans_data = scaler.transform(prediction_data)
            
            # 转换为 LNN 格式 [Batch, Seq, Features]
            tensor_data = torch.tensor(trans_data, dtype=torch.float32)
            feature_size = tensor_data.shape[1] // self.seq_length
            seriesed_data = tensor_data.view(-1, self.seq_length, feature_size)
            
            return seriesed_data.to(DEVICE), col_name
        except Exception as e:
            st.error(f"数据处理异常: {e}")
            return None, None

# ========== 6. 主程序界面逻辑 ==========
check_disclaimer()

# --- 醒目的标题区 ---
st.markdown('<p class="main-title">🌽 玉米储存年份预测分析系统</p >', unsafe_allow_html=True)
st.markdown("""
    <p class="sub-title">
        <span class="interface-tag">主界面</span> 
        基于液态神经网络 (LNN) 的快速无损检测方案
    </p >
""", unsafe_allow_html=True)

scaler, model = load_resources()

if scaler and model:
    # 侧边栏配置
    with st.sidebar:
        st.markdown("<h2 style='color:white;'>🛠️ 系统控制台</h2>", unsafe_allow_html=True)
        st.write("---")
        if st.checkbox("查看审计日志"):
            if os.path.exists(LOG_FILE_PATH):
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    st.download_button("📥 导出点击记录", f.read(), file_name="audit_log.txt")
        st.markdown("<br><br><p style='color:#475569;'>Version 3.0 (Apple Style)</p >", unsafe_allow_html=True)

    # 左右布局：左边上传，右边预测结果
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown('<div class="vibrant-card">', unsafe_allow_html=True)
        st.markdown("#### 📂 数据载入")
        file = st.file_uploader("支持 Excel 或 CSV 格式", type=["xlsx", "csv"], label_visibility="collapsed")
        
        if file:
            # 自动识别格式
            df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
            st.info(f"📄 文件已就绪: {len(df)} 个样本")
            
            # 数据预览
            with st.expander("点击预览原始数据"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 执行自动化预测"):
                st.session_state.start_inference = True
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if file and 'start_inference' in st.session_state:
            with st.spinner("⚡ 核心引擎正在处理中..."):
                processor = Data_prepossing(SEQ_LENGTH=1)
                input_tensor, names = processor.prediction_pretreatment(df, scaler)

                if input_tensor is not None:
                    # 模型推理
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        confs = torch.max(probs, dim=1)[0]

                    st.markdown("#### 📊 智能分析报告")
                    labels = ['≤1年 (新粮)', '1-2年', '2-3年', '3年及以上 (陈粮)']
                    card_styles = ['res-card-0', 'res-card-1', 'res-card-2', 'res-card-3']

                    for name, p_idx, conf in zip(names, preds, confs):
                        idx = int(p_idx)
                        style = card_styles[idx]
                        
                        # 展示彩色结果卡片
                        st.markdown(f"""
                            <div class="vibrant-card {style}" style="padding: 18px 25px; margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="flex:1;">
                                        <div style="color: #64748b; font-size: 13px; font-weight: 400;">样品标识</div>
                                        <div style="font-size: 18px; font-weight: 700; color: #1e293b;">{name}</div>
                                    </div>
                                    <div style="flex:1; text-align: center;">
                                        <div style="color: #64748b; font-size: 13px; font-weight: 400;">预测年份</div>
                                        <div style="font-size: 22px; font-weight: 700; color: #1e3a8a;">{labels[idx]}</div>
                                    </div>
                                    <div style="flex:1; text-align: right;">
                                        <div style="color: #64748b; font-size: 13px; font-weight: 400;">判定置信度</div>
                                        <div style="font-size: 22px; font-weight: 700; color: #334155;">{conf:.1%}</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # 简单的信心建议
                        if conf < 0.6:
                            st.caption(f"💡 样品 {name} 置信度较低，建议结合理化实验进一步确认。")
        else:
            st.markdown("""
                <div style="text-align: center; padding: 80px; color: #cbd5e1; border: 2px dashed #e2e8f0; border-radius: 20px;">
                    <p style="font-size: 50px;">📉</p >
                    <p>待分析数据已就绪，请点击左侧按钮开始</p >
                </div>
            """, unsafe_allow_html=True)
