from speech_detector import SpeechDetector
from fireredasr.models.fireredasr_streaming import FireRedAsrStreaming
import numpy as np
import time


class RealtimeSpeechRecognizer:
    def __init__(
        self,
        model_dir="pretrained_models",
        use_gpu=False,
        sample_rate=16000,
        silence_duration_s=0.4,
        transcribe_interval=1.0,
    ):
        self.model = FireRedAsrStreaming(model_dir,
                                         use_gpu=use_gpu,
                                         sample_rate=sample_rate)
        self.detector = SpeechDetector(model_dir,
                                       framerate=sample_rate,
                                       silence_duration_s=silence_duration_s)
        self.sample_rate = sample_rate
        self.transcribe_interval = transcribe_interval
        self.sentence_id = 0
        self.speech_state = False
        self.sample_count = 0
        self.next_transcribe_time = 0.0

    def gen_result(self, t, text=None, latency=0.0):
        return {
            "type": t,
            "id": self.sentence_id,

            "text": text,
            "ts": self.sample_count/self.sample_rate,
            "latency": latency,
        }

    def transcribe(self):
        start_time = time.time()
        text = self.model.transcribe()
        return text, time.time() - start_time

    def recognize(self, audio_bytes):
        results = []
        wav_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        for frame_np, is_speech in self.detector.detect(wav_np):
            if is_speech:
                self.model.input(frame_np)
            if is_speech and not self.speech_state:
                results.append(self.gen_result("begin"))
                self.next_transcribe_time = self.transcribe_interval
            elif self.speech_state and not is_speech:
                text, cost = self.transcribe()
                results.append(self.gen_result("end", text, cost))
                self.model.clear_state()
                self.sentence_id += 1
            elif self.speech_state:
                cur_ts = self.model.get_input_length() / self.sample_rate
                if cur_ts >= self.next_transcribe_time:
                    text, cost = self.transcribe()
                    results.append(self.gen_result("changed", text, cost))
                    self.next_transcribe_time = cur_ts + self.transcribe_inter
            self.speech_state = is_speech
            self.sample_count += len(frame_np)
        return results
