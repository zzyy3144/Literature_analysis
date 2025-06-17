import streamlit as st
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
import joblib
import numpy as np
import jieba
import re
import time  # 导入time模块用于计时

# 设置页面配置
st.set_page_config(page_title="智能科研文献分析系统", page_icon="📚", layout="wide")

# 数据预处理函数
@st.cache_data
def preprocess_text(text):
    try:
        text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
        text = re.sub(r'[^\w\s]', '', text)  # 去除特殊字符
        words = jieba.cut(text)  # 中文分词
        return ' '.join(words)
    except Exception as e:
        st.error(f"文本预处理时出错: {e}")
        return text

# 将文本转换为模型输入的张量
@st.cache_data
def convert_to_tensor(texts, tokenizer, max_seq_length=128):
    try:
        inputs = tokenizer(texts, max_seq_len=max_seq_length, pad_to_max_seq_len=True)
        input_ids = paddle.to_tensor([inputs['input_ids']])
        token_type_ids = paddle.to_tensor([inputs['token_type_ids']])
        return input_ids, token_type_ids
    except Exception as e:
        st.error(f"文本转换为张量时出错: {e}")
        return None, None

# 加载模型和标签编码器
@st.cache_resource
def load_models():
    try:
        start_time = time.time()  # 记录开始时间
        dt_model = joblib.load('dt_model.pkl')
        ernie_model_state = paddle.load('ernie_model.pdparams')
        ernie_sentiment_model_state = paddle.load('ernie_sentiment_model.pdparams')
        label_encoder_category = joblib.load('label_encoder_category.pkl')
        label_encoder_sentiment = joblib.load('label_encoder_sentiment.pkl')
        end_time = time.time()  # 记录结束时间
        st.write(f"模型和标签编码器加载成功，耗时 {end_time - start_time:.2f} 秒")
        return dt_model, ernie_model_state, ernie_sentiment_model_state, label_encoder_category, label_encoder_sentiment
    except Exception as e:
        st.error(f"加载模型或标签编码器时出错: {e}")
        st.stop()

# 初始化模型
@st.cache_resource
def initialize_models(ernie_model_state, ernie_sentiment_model_state):
    try:
        start_time = time.time()  # 记录开始时间
        tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
        model = ErnieForSequenceClassification.from_pretrained('ernie-1.0', num_classes=2)
        sentiment_model = ErnieForSequenceClassification.from_pretrained('ernie-1.0', num_classes=3)
        model.set_state_dict(ernie_model_state)
        sentiment_model.set_state_dict(ernie_sentiment_model_state)
        end_time = time.time()  # 记录结束时间
        st.write(f"模型初始化成功，耗时 {end_time - start_time:.2f} 秒")
        return tokenizer, model, sentiment_model
    except Exception as e:
        st.error(f"初始化模型时出错: {e}")
        st.stop()

# 主函数
def main():
    st.title("智能科研文献分析系统")
    st.markdown("欢迎使用智能科研文献分析系统！请输入文献摘要进行分类和情感分析。")

    # 用户输入
    with st.container():
        text = st.text_area("文献摘要", "", height=150)

    if st.button("分析"):
        if text:
            try:
                with st.spinner('正在分析...'):
                    # 加载模型和标签编码器
                    dt_model, ernie_model_state, ernie_sentiment_model_state, label_encoder_category, label_encoder_sentiment = load_models()
                    
                    # 初始化模型
                    tokenizer, model, sentiment_model = initialize_models(ernie_model_state, ernie_sentiment_model_state)
                    
                    # 文本预处理
                    start_time = time.time()  # 记录开始时间
                    processed_text = preprocess_text(text)
                    end_time = time.time()  # 记录结束时间
                    st.write(f"文本预处理完成，耗时 {end_time - start_time:.2f} 秒")
                    
                    # 转换为张量
                    start_time = time.time()  # 记录开始时间
                    input_ids, token_type_ids = convert_to_tensor([processed_text], tokenizer)
                    end_time = time.time()  # 记录结束时间
                    st.write(f"张量转换完成，耗时 {end_time - start_time:.2f} 秒")
                    
                    # 文本分类
                    model.eval()
                    start_time = time.time()  # 记录开始时间
                    category_logits = model(input_ids, token_type_ids=token_type_ids)
                    category_pred = paddle.argmax(category_logits, axis=1).numpy()[0]
                    category_label = label_encoder_category.inverse_transform([category_pred])[0]
                    end_time = time.time()  # 记录结束时间
                    st.write(f"文本分类完成，耗时 {end_time - start_time:.2f} 秒")
                    
                    # 情感分析
                    sentiment_model.eval()
                    start_time = time.time()  # 记录开始时间
                    sentiment_logits = sentiment_model(input_ids, token_type_ids=token_type_ids)
                    sentiment_pred = paddle.argmax(sentiment_logits, axis=1).numpy()[0]
                    sentiment_label = label_encoder_sentiment.inverse_transform([sentiment_pred])[0]
                    end_time = time.time()  # 记录结束时间
                    st.write(f"情感分析完成，耗时 {end_time - start_time:.2f} 秒")
                    
                    # 显示结果
                    with st.container():
                        st.write(f"**分类结果:** {category_label}")
                        st.write(f"**情感分析结果:** {sentiment_label}")
            except Exception as e:
                st.error(f"分析过程中出错: {e}")
        else:
            st.warning("请输入文献摘要！")

if __name__ == "__main__":
    main()