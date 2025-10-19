import cv2
import numpy as np
import base64
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from io import BytesIO
import json
import streamlit as st

class KTS:
    def __init__(self, kernel, num_segments):
        self.kernel = kernel
        self.num_segments = num_segments

    def predict(self):
        N = self.kernel.shape[0]
        costs = np.full((N + 1, self.num_segments + 1), float('inf'))
        split_points = np.zeros((N + 1, self.num_segments + 1), dtype=int)
        
        costs[0, 0] = 0

        for n in range(1, N + 1):
            for k in range(1, self.num_segments + 1):
                min_cost = float('inf')
                best_split = 0
                for t in range(k, n + 1):
                    sub_kernel = self.kernel[t-1:n, t-1:n]
                    cost = -np.sum(np.diag(sub_kernel))
                    
                    current_cost = costs[t - 1, k - 1] + cost
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_split = t - 1
                
                costs[n, k] = min_cost
                split_points[n, k] = best_split

        boundaries = []
        current_split = N
        for k in range(self.num_segments, 0, -1):
            boundaries.append(current_split)
            current_split = split_points[current_split, k]
        
        return sorted([0] + boundaries)

class VideoProcessor:
    def __init__(self):
        @st.cache_resource
        def get_models():
            print("正在加载模型。")
            vision_model = SentenceTransformer('clip-ViT-B-32')
            llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
            print("模型加载完成。")
            return vision_model, llm
        
        self.vision_model, self.llm = get_models()

    def _encode_image_to_base64(self, image_array):
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @st.cache_data(show_spinner=False)
    def process_video(_self, video_bytes, sampling_rate, num_segments_kts):
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        with st.spinner("阶段 1/4: 正在采样视频帧并提取语义特征..."):
            frames, features = _self._extract_semantic_features(temp_video_path, sampling_rate)
            if not frames:
                st.error("无法从视频中提取帧。请检查视频文件。")
                return None
        
        with st.spinner(f"阶段 2/4: 正在使用KTS将视频分割为 {num_segments_kts} 个片段..."):
            segments = _self._identify_event_segments_kts(features, num_segments_kts)
            st.success(f"视频被成功分割成 {len(segments)} 个事件片段。")

        with st.spinner("阶段 3/4: 正在聚类片段以发现宏观场景..."):
            scene_clusters, segment_features = _self._discover_scenes(segments, features)
            num_scenes = len(set(scene_clusters.keys())) - (1 if -1 in scene_clusters else 0)
            st.success(f"发现了 {num_scenes} 个不同的宏观场景。")

        with st.spinner("阶段 4/4: 正在为每个场景生成关键帧和高层特征..."):
            summary_data = _self._generate_summary(scene_clusters, segments, frames, features, segment_features)
            st.success("视频处理完毕！")
        
        os.remove(temp_video_path)
        return summary_data

    def _extract_semantic_features(self, video_path, sampling_rate):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sampling_rate) if sampling_rate > 0 else 1
        
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        
        if not frames: return [], []

        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        feature_vectors = self.vision_model.encode(pil_images, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return frames, feature_vectors

    def _identify_event_segments_kts(self, features, num_segments):
        similarity_matrix = cosine_similarity(features)
        kts = KTS(kernel=similarity_matrix, num_segments=num_segments)
        boundaries = kts.predict()
        
        segments = []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i+1]
            if start_idx < end_idx:
                segments.append((start_idx, end_idx))
        return segments

    def _discover_scenes(self, segments, features):
        segment_features = np.array([np.mean(features[start:end], axis=0) for start, end in segments])
        
        # DBSCAN参数可以根据需要调整
        dbscan = DBSCAN(eps=0.2, min_samples=2, metric='cosine')
        cluster_labels = dbscan.fit_predict(segment_features)
        
        scene_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in scene_clusters: scene_clusters[label] = []
            scene_clusters[label].append(i)
            
        return scene_clusters, segment_features

    def _generate_summary(self, scene_clusters, segments, frames, features, segment_features):
        summary_data = {}
        for scene_id, segment_indices in scene_clusters.items():
            if scene_id == -1: continue

            cluster_feature_vectors = segment_features[segment_indices]
            cluster_centroid = np.mean(cluster_feature_vectors, axis=0)
            
            # 找到最接近质心的片段
            distances = [np.linalg.norm(vec - cluster_centroid) for vec in cluster_feature_vectors]
            closest_segment_in_cluster_idx = np.argmin(distances)
            original_segment_idx = segment_indices[closest_segment_in_cluster_idx]
            
            # 在该片段中找到最接近片段均值的帧作为关键帧
            start_idx, end_idx = segments[original_segment_idx]
            keyframe_idx = start_idx + np.argmax([cosine_similarity(f.reshape(1,-1), segment_features[original_segment_idx].reshape(1,-1)) for f in features[start_idx:end_idx]])

            keyframe_image = frames[keyframe_idx]
            base64_image = self._encode_image_to_base64(keyframe_image)
            
            prompt_text = "你是一个先进的多模态AI助手。请仔细观察以下图片，并用一句话简洁地概括这个场景的核心内容。"
            message = HumanMessage(content=[{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
            response = self.llm.invoke([message])
            description = response.content

            summary_data[f"场景 {scene_id}"] = {
                "description": description,
                "keyframe_base64": base64_image,
                "segment_count": len(segment_indices)
            }
        return summary_data

    def answer_question(self, question, video_summary):
        system_prompt = f"""
        你是一个智能视频分析助手，一个Video LLM。你刚刚完整地观看并分析了一段用户上传的视频。
        你已经将视频内容分解成了几个关键的宏观场景，并生成了摘要。这是你对视频内容的内部知识库：

        --- 视频内容摘要 ---
        {json.dumps(video_summary, ensure_ascii=False, indent=2)}
        --- 摘要结束 ---

        现在，用户要向你提问。请严格基于你脑中的这份摘要来回答问题。
        以第一人称（例如，“我看到的视频...”，“在这个视频里...”）进行回答，让用户感觉你真的看过了视频。
        如果问题超出了摘要的范围，请诚实地回答你无法从视频中获取该信息。
        保持回答简洁、直接。
        """

        full_prompt = f"{system_prompt}\n\n用户的问题是：{question}"
        
        response = self.llm.invoke(full_prompt)
        return response.content
