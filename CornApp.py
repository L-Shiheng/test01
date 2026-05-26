import os
import csv
import joblib
import torch
import pandas as pd
import streamlit as st
import datetime
import warnings
from typing import Union
from io import StringIO,BytesIO

warnings.filterwarnings('ignore')

# ========== 1. 路径与设备配置 ==========
# 如果部署到 GitHub，建议改为相对路径，例如：IMPUTER_SCALER_PATH = 'corn_treat.pkl'
IMPUTER_SCALER_PATH = 'D:\\AIOnline\\corn\\corn_treat.pkl'   
LNN_MODEL_PATH = 'D:\\AIOnline\\corn\\LNNclassification.pt'     
LOG_FILE_PATH = 'D:\\AIOnline\\corn\\user_agreement_log.txt'
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
        prediction = uploaded_file.iloc[:,0:-2].values

        # 获取样品名称
        col_name = uploaded_file.index.tolist()

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

# ==========================================
# 5. 数据文件处理
# ==========================================
column_names = ['2-Butanone', '2-Ethylfuran', 'Diacetyl', 'Dimethyl disulfide', 'Hexanal', 'Mesityl oxide', '1-Butanol', 'Methyl hexanoate',
                'Isoamyl alcohol','2-Methylpyridine', 'trans-2-Hexenal', '2-Pentylfuran', '2-Methylpyrazine', '2-Ethylpyridine', '2-Octanone',
                'Octanal', '1-Octen-3-one', '2,5-Dimethylpyrazine', 'trans-2-Heptenal', '6-Methyl-5-hepten-2-one', '1-Hexanol', 'Dimethyl trisulfide',
                'Methyl octanoate', 'Nonanal', '2,3,5-Trimethylpyrazine', 'trans-3-Octen-2-one', 'trans-2-Octenal', '(Z)-Linalool oxide', '1-Octen-3-ol',
                '1-Heptanol', 'Acetic acid', 'Furfural', '(E)-Linalool oxide', '2-Ethylhexanol', 'n-Decanal', '2-Acetylfuran', 'Benzaldehyde',
                'Ethyl nonanoate', '(E)-2-Nonenal', 'Propanoic acid', '1-Octanol', 'Isobutyric acid', '2-Pentylpyridine', '5-Methyl furfural',
                '(2E,6Z)-nona-2,6-dienal', '2-Undecanone', 'Undecanal', '(E)-2-Octen-1-ol', 'Benzonitrile', 'Butyric acid', 'trans-2-Decenal', 
                'Phenylacetaldehyde', 'Acetophenone', '1-Nonanol', 'Furfuryl alcohol', 'Isovaleric acid', '(E,E)-2,4-nonadienal', 'gamma-Caprolactone', 
                'n-Dodecanal', 'Valeric acid', 'trans-2-Undecenal', '1-Decanol', 'p-Acetyltoluene', 'Methyl salicylate', 'Cuminaldehyde', 
                'gamma-Heptalactone', '2,4-Decadienal', 'Capronic acid', 'trans-Geranylacetone', 'Guaiacol', 'Butyl benzoate', 'Benzyl alcohol', 
                '2-Phenylethanol', 'gamma-Octalactone', 'beta-Ionone', '1-Dodecanol', 'Benzothiazole', '2-Acetylpyrrole', 'o-Cresol', 'Phenol', 
                '4-Ethyl-2-methoxyphenol', '4-Methoxybenzaldehyde', 'gamma-Nonalactone', 'Pantolactone', 'trans-Cinnamaldehyde', 'Caprylic acid', 
                'p-Cresol', 'Hexadecanal', 'gamma-Decalactone', 'Nonanoic acid', '4-Ethylphenol', 'p-Vinylguaiacol', 'o-Aminoacetophenone', 
                'Massoia lactone', 'Capric acid', 'Indole', 'Skatole', 'Vanillin']

def detect_encoding_from_bytes(data: bytes) -> str:
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'latin1']
    for enc in encodings:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return 'latin1'

def extract_compound_data(file_input: Union[str, BytesIO], output_csv=None):
    """
    智能解析文件：.xlsx 直接读取；.csv 按 GC-MS 特殊格式提取。
    参数 file_input 可以是文件路径字符串，或 BytesIO / UploadedFile 对象。
    """
    # 0. 类型保护：禁止传入布尔值、None 等
    if file_input is None:
        raise ValueError("文件输入为空，请检查是否已上传文件")
    if isinstance(file_input, bool):
        raise TypeError("file_input 不能为布尔值，请传入有效的文件路径或上传文件对象")

    # 获取文件名（用于判断扩展名）
    if isinstance(file_input, str):
        filename = file_input
    else:
        filename = getattr(file_input, 'name', '')

    # ------------------- 处理 Excel 文件 -------------------
    if filename.lower().endswith('.xlsx'):
        if isinstance(file_input, str):
            df = pd.read_excel(file_input)
        else:
            file_input.seek(0)
            df = pd.read_excel(file_input)
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return df

    # ------------------- 处理 CSV 文件（GC-MS 特殊格式） -------------------
    # 读取内容并自动检测编码
    if isinstance(file_input, str):
        encoding = detect_encoding_from_bytes(open(file_input, 'rb').read(1024))
        with open(file_input, 'r', encoding=encoding) as f:
            content = f.read()
    else:
        file_input.seek(0)
        raw_data = file_input.read()
        encoding = detect_encoding_from_bytes(raw_data)
        content = raw_data.decode(encoding)

    f = StringIO(content)
    reader = csv.reader(f)

    sample_names = []
    compound_data = {}

    # 1. 定位到“数据”表头行
    #len(row) >= 2 and (row[0] == '数据' or row[0] == 'Data') and (row[1] == '数据文件路径' or row[1] == 'Data File Path'):
    for row in reader:
        if len(row) >= 2 and (row[0] == '数据' or row[0] == 'Data') and (row[1] == '数据文件路径' or row[1] == 'Data File Path'):
            break
    else:
        raise ValueError("未找到数据表头（'Data','Data File Path'）")
        #ValueError("未找到数据表头（'数据','数据文件路径'）") or ValueError("未找到数据表头行（'Data','Data File Path'）")

    # 2. 读取所有数据行（每个样品一行）
    data_rows = []
    for row in reader:
        if not row:
            continue
        if (row[0].startswith('数据') or row[0].startswith('Data')) and ':' in row[0]:
            data_rows.append(row)
        elif row[0] == '[结果](峰面积)' or row[0] == '[Result](Area)':
            break

    # 3. 从数据文件路径提取样品名称（去掉.qgd）
    for idx, row in enumerate(data_rows, start=1):
        if len(row) >= 2:
            file_path = row[1].strip()
            base_name = os.path.basename(file_path)
            if base_name.lower().endswith('.qgd'):
                sample_name = base_name[:-4]
            else:
                sample_name = base_name
            if not sample_name:
                sample_name = f"Sample_{idx}"
        else:
            sample_name = f"Sample_{idx}"
        sample_names.append(sample_name)

    # 4. 读取结果表头
    # header_row = next(reader)   # ["ID", "组分名称", "数据1 峰面积", ...]

    # 5. 读取每个化合物的峰面积数据
    for row in reader:
        if not row or len(row) < 2:
            continue
        compound_name = row[1].strip()
        areas = []
        for val in row[2:len(sample_names)+2]:  # 只读取与样品数量对应的列
            try:
                clean_val = val.strip().replace(',', '')
                areas.append(float(clean_val) if clean_val else 0.0)
            except ValueError:
                areas.append(0.0)
        # if len(areas) < len(sample_names):
        #     areas.extend([0.0] * (len(sample_names) - len(areas)))
        compound_data[compound_name] = areas

    # 构建 DataFrame：行=化合物，列=样品
    df = pd.DataFrame(compound_data, index=sample_names)

    if output_csv:
        df.to_csv(output_csv, encoding='utf-8-sig')
        print(f"数据已保存至: {output_csv}")
        
    return df

# ==========================================
# 5. 主程序界面
# ==========================================
# 权限检查
check_disclaimer()
# 主界面
st.title("🌽 玉米储存年份预测系统")

# 清除缓存，确保每次加载最新资源
st.cache_resource.clear()

# 上载模型和预处理器
uploaded_file = st.file_uploader("上传样品的CSV文件", type=["csv"], accept_multiple_files=False)

if uploaded_file:
    if st.button("开始预测"):
        # 数据提取与预处理
        prepossessor = Data_prepossing(SEQ_LENGTH=1,SEQ_SIZE=80)
        df_extract= extract_compound_data(uploaded_file)  # 智能解析上传的文件，得到 DataFrame
        df_process = df_extract[column_names]  # 确保样品名称在第一列，化合物数据在后续列

        # 数据展示
        st.write("### 数据预览")
        st.dataframe(df_process.head())

        # 数据预处理
        input_tensor, col_name = prepossessor.prediction_pretreatment(df_process)
        
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