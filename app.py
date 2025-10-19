import streamlit as st
from video_processor import VideoProcessor
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="Xiaoice | 视频对话", layout="wide", page_icon="🎬")

st.title("🎬 Xiaoice Video LLM: 与你的视频对话")
st.markdown("""
欢迎使用 Xiaoice，这是一个基于论文《通过自监督时空语义特征聚类实现免训练的视频理解》的交互式应用。

**工作流程:**
1.  **上传视频**: 在左侧上传一个视频文件 (建议使用1-3分钟的 .mp4 文件以获得最佳体验)。
2.  **自动分析**: AI将自动处理视频，提取特征，分割事件，发现场景，并生成摘要。
3.  **开始对话**: 分析完成后，您可以在下方的对话框中就视频内容向AI提问。
""")

if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()
if 'video_summary' not in st.session_state:
    st.session_state.video_summary = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

processor = st.session_state.processor

with st.sidebar:
    st.header("⚙️ 设置")
    uploaded_file = st.file_uploader("上传你的视频文件", type=["mp4", "mov", "avi"])
    
    st.subheader("分析参数")
    sampling_rate = st.slider("采样率 (帧/秒)", 1, 5, 2, help="从视频中每秒提取多少帧进行分析。更高的值会更精确但更慢。")
    num_segments_kts = st.slider("KTS期望片段数", 5, 50, 15, help="期望将视频分割成多少个基本事件片段。这会影响场景发现的粒度。")

    if st.button("🚀 开始分析视频", use_container_width=True, disabled=not uploaded_file):
        st.session_state.video_summary = None
        st.session_state.messages = []
        
        video_bytes = uploaded_file.getvalue()
        with st.status("正在处理视频，请稍候...", expanded=True) as status:
            st.session_state.video_summary = processor.process_video(video_bytes, sampling_rate, num_segments_kts)
            status.update(label="视频处理完成！", state="complete", expanded=False)

st.header("📊 视频摘要")

if st.session_state.video_summary:
    for scene_name, scene_data in st.session_state.video_summary.items():
        with st.expander(f"**{scene_name}**: {scene_data['description']}", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                img_data = base64.b64decode(scene_data['keyframe_base64'])
                img = Image.open(BytesIO(img_data))
                st.image(img, caption=f"{scene_name} 的代表性关键帧", use_column_width=True)
            with col2:
                st.write(f"**AI 生成的 Overview:**")
                st.info(scene_data['description'])
                st.write(f"**构成:** 此场景由 **{scene_data['segment_count']}** 个相似的事件片段组成。")
else:
    st.info("请在左侧上传视频并点击“开始分析视频”以查看结果。")

st.divider()

st.header("💬 与视频对话")

if st.session_state.video_summary:
    # 显示历史对话消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input("请就视频内容向我提问..."):
        # 将用户消息添加到历史记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成并显示AI的回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = processor.answer_question(prompt, st.session_state.video_summary)
                st.markdown(response)
        
        # 将AI的回答添加到历史记录
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("完成视频分析后，您可以在这里开始对话。")
