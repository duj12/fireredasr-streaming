import asyncio
import uuid
import time
import numpy as np
from typing import Dict, Any
from speech_detector import SpeechDetector
from fireredasr.models.fireredasr_streaming import FireRedAsrStreaming


class AsyncRealtimeSpeechRecognizer:
    def __init__(
        self,
        model_dir="pretrained_models",
        use_gpu=True,
        sample_rate=16000,
        silence_duration_s=0.4,
        transcribe_interval=1.0,
    ):
        # 共享模型
        self.shared_model = FireRedAsrStreaming(
            model_dir, use_gpu=use_gpu, sample_rate=sample_rate
        )
        self.sample_rate = sample_rate
        self.silence_duration_s = silence_duration_s
        self.transcribe_interval = transcribe_interval

        # 每个session独立缓存
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # 模型锁，确保同一时间只有一个线程调用 shared_model.transcribe()
        self.model_lock = asyncio.Lock()

    def create_session(self):
        """为每个并发请求创建独立状态"""
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "detector": SpeechDetector(
                framerate=self.sample_rate,
                silence_duration_s=self.silence_duration_s,
            ),
            "speech_state": False,
            "audio_buffer": np.empty(0),
            "asr_state": None,   # asr临时状态  
            "sample_count": 0,
            "next_transcribe_time": 0.0,
            "sentence_id": 0,
        }
        return sid

    def gen_result(self, sid, t, text=None, latency=0.0):
        s = self.sessions[sid]
        return {
            "type": t,
            "id": s["sentence_id"],
            "text": text,
            "ts": s["sample_count"] / self.sample_rate,
            "latency": latency,
            "session": sid,
        }

    async def recognize(self, sid: str, audio_bytes: bytes):
        """异步处理单个session的音频流"""
        s = self.sessions[sid]
        results = []

        wav_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        for frame_np, is_speech in s["detector"].detect(wav_np):
            if is_speech:
                s["audio_buffer"] = np.concatenate((s["audio_buffer"], frame_np))

            if is_speech and not s["speech_state"]:
                results.append(self.gen_result(sid, "begin"))
                s["next_transcribe_time"] = self.transcribe_interval

            elif s["speech_state"] and not is_speech:
                # 检测到一句结束
                text, cost = await self.transcribe_from_model(sid)
                results.append(self.gen_result(sid, "end", text, cost))
                self.clear_session_buffer(sid)

            elif s["speech_state"]:
                cur_ts = len(s["audio_buffer"]) / self.sample_rate
                if cur_ts >= s["next_transcribe_time"]:
                    text, cost = await self.transcribe_from_model(sid)
                    results.append(self.gen_result(sid, "changed", text, cost))
                    s["next_transcribe_time"] = cur_ts + self.transcribe_interval

            s["speech_state"] = is_speech
            s["sample_count"] += len(frame_np)

        return results       

    async def transcribe_from_model(self, sid):
        """线程安全的转录调用"""
        async with self.model_lock:
            start_time = time.time()
            text, asr_state = self.shared_model.transcribe(self.sessions[sid]["audio_buffer"], self.sessions[sid]["asr_state"])
            self.sessions[sid]["asr_state"] = asr_state  # 更新asr状态
            latency = time.time() - start_time
        return text, latency

    def clear_session_buffer(self, sid):
        """清空单个session缓存"""
        s = self.sessions[sid]
        s["audio_buffer"] = np.empty(0)
        s['asr_state'] = None
        s["sentence_id"] += 1

