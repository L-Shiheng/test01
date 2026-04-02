import os
import datetime
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 定义设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 路径配置 ==========
# 请确保这些路径与你电脑上的实际存放位置一致
IMPUTER_SCALER_PATH = 'corn_treat.pkl'       
LNN_MODEL_PATH = 'LNNclassification.pt'     
LOG_FILE_PATH = 'user_agreement_log.txt'     

# ==========================================
# 1. 免责声明与合规记录模块
# ==========================================
def check_disclaimer_and_log():
    # 初始化 session_state 来记录用户是否已经同意
    if 'agreed_to_disclaimer' not in st.session_state:
        st.session_state.agreed_to_disclaimer = False

    # 如果用户还没同意，就展示免责声明，并阻断后续渲染
    if not st.session_state.agreed_to_disclaimer:
        st.title("🌽 农产品/玉米年份预测系统")
        st.warning("⚠️ 在进入系统前，请仔细阅读以下免责声明：")
        
        # 使用 info 框展示免责声明正文
        st.info("""
        **免责声明**
        
        本内容由人工智能模型生成，所呈现的预测、分析或结论仅为基于已有数据与算法的推演结果，不构成任何形式的保证、承诺或专业建议。AI模型可能存在误差、偏差或对未知因素的考虑不足，实际结果可能与预测存在显著差异。用户应结合自身判断、专业咨询及实时信息独立做出决策，并自行承担因使用本预测内容所引发的一切风险与责任。开发者及提供方不对任何直接或间接损失承担责任。
        
        **请谨慎使用预测结果，切勿将其作为唯一决策依据。**
        """)
        
        # 确认按钮
        if st.button("我已充分阅读并同意上述免责声明，开始使用系统", type="primary"):
            # --- 核心：将同意行为写入后台日志文件 ---
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 记录时间、操作。如果有用户登录系统，这里还可以加上用户名或IP
            log_entry = f"[{current_time}] 操作：用户点击同意免责声明并进入系统。\n"
            
            try:
                # 以追加模式 'a' 打开/创建日志文件
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                st.toast(f"后台日志记录异常（不影响使用）: {e}")

            # 记录状态为已同意，并重新刷新页面
            st.session_state.agreed_to_disclaimer = True
            st.rerun()
            
        # st.stop() 会立刻终止代码运行，所以下面的界面在同意前不会显示
        st.stop()

# ==========================================
# 2. 资源加载模块
# ==========================================
@st.cache_resource
def load_preprocessor():
    if not os.path.exists(IMPUTER_SCALER_PATH):
        st.error(f"❌ 找不到预处理文件：{IMPUTER_SCALER_PATH}")
        return None
    try:
        # 你的 pkl 包含 Pipeline(KNNImputer + Normalizer)
        scaler = joblib.load(IMPUTER_SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"预处理器加载失败: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(LNN_MODEL_PATH):
        st.error(f"❌ 找不到模型文件：{LNN_MODEL_PATH}")
        return None
    try:
        # 根据上传的 pt 文件结构，这是 torch.jit.save 导出的 TorchScript 模型
        # 使用 torch.jit.load 可以直接加载，不需要在代码里写复杂的 Class 定义！
        model = torch.jit.load(LNN_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return model
    except Exception as e:
        # 作为备选，如果不是 TorchScript，尝试普通的 load
        try:
            model = torch.load(LNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.eval()
            return model
        except Exception as e2:
            st.error(f"模型加载失败: 主错误 {e}, 备选错误 {e2}")
            return None

# ==========================================
# 3. 主界面与预测逻辑 (同意声明后才执行)
# ==========================================
# 首先执行免责声明检查
check_disclaimer_and_log()

# 用户同意后，正式进入系统界面
st.title("🌽 农产品/玉米年份预测系统")

# 加载模型
preprocessor = load_preprocessor()
model = load_model()

if model and preprocessor:
    st.success("✅ 模型和数据预处理引擎已加载就绪。")
    
    # 文件上传
    uploaded_file = st.file_uploader("请上传待预测的样本数据 (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.write("### 数据预览")
            st.dataframe(df.head())
            
            if st.button("开始批量预测"):
                with st.spinner("正在进行数据标准化和推理运算..."):
                    # 假设第一列是样品名称，后面的列是特征
                    col_name = df.iloc[:, 0].values  
                    features = df.iloc[:, 1:]        
                    
                    # 1. 预处理 (KNN Imputer -> Normalizer)
                    processed_features = preprocessor.transform(features)
                    
                    # 2. 转为 Tensor
                    # 注意：如果你的模型需要 3D 输入 (Batch, Seq_len, Features)，可能需要 unsqueeze
                    # 这里假设模型支持批量 2D 推理，如果不兼容，可以改为 input_tensor.unsqueeze(1) 等
                    input_tensor = torch.from_numpy(processed_features.astype(np.float32)).to(DEVICE)
                    
                    # 3. 推理预测
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_classes = torch.argmax(probabilities, dim=1)
                        confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
                    
                    # 4. 结果展示
                    st.divider()
                    st.subheader("🔮 分析报告")
                    
                    group_name = ['≤1 year', '1-2 year', '2-3 year', '3+ year']
                    
                    # 遍历并展示每个样本的结果
                    for sample_idx, (name, p_class, conf) in enumerate(zip(col_name, predicted_classes, confidences)):
                        class_idx = int(p_class.item())
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"样本 #{sample_idx+1}", f"{name}")
                        col2.metric("预测类别", f"{group_name[class_idx]}")
                        col3.metric("置信度", f"{conf:.1%}")
                        
                        if conf > 0.8:
                            st.success("模型对此结果非常有信心。")
                        elif conf > 0.6:
                            st.warning("模型具有一定信心，建议复核。")
                        else:
                            st.error("置信度较低，结果仅供参考。")
                        st.write("---")
                        
        except Exception as e:
            st.error(f"处理数据时发生错误：{e}")
