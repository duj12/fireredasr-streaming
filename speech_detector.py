import numpy as np
import vad as vad


class SpeechDetector:

    def __init__(self,
                 model_dir="pretrained_models",
                 framerate=16000,
                 threshold=0.5,
                 silence_duration_s=0.8,
                 max_speech_duration_s=30,
                 ):
        self.framerate = framerate
        self.threshold = threshold
        self.silence_duration_s = silence_duration_s
        self.max_speech_duration_s = max_speech_duration_s

        self.model = vad.get_vad_model(model_dir)
        self.state = self.model.get_initial_state(batch_size=1)
        self.audio_buffer = None
        self.silence_last_s = 0
        self.is_speech = False
        self.samples_count = 0
        self.last_speech_pos = 0

    def detect(self, audio):
        samples = 512 if self.framerate == 16000 else 256
        det_interval_s = float(samples) / self.framerate
        if self.audio_buffer is None:
            self.audio_buffer = audio
        else:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio))

        neg_threshold = max(self.threshold - 0.15, 0.01)
        while len(self.audio_buffer) >= samples:
            audio = self.audio_buffer[:samples]
            speech_prob, self.state = self.model(
                audio / 32768.0, self.state, self.framerate)
            speech_threshold = neg_threshold if self.is_speech else self.threshold
            if speech_prob > speech_threshold:
                self.silence_last_s = 0
                if not self.is_speech:
                    self.is_speech = True
                    self.last_speech_pos = self.samples_count
                else:
                    speech_frames = self.samples_count - self.last_speech_pos
                    if speech_frames / self.framerate > self.max_speech_duration_s:
                        self.is_speech = False
            else:
                if self.is_speech:
                    self.silence_last_s += det_interval_s
                    if self.silence_last_s >= self.silence_duration_s:
                        self.is_speech = False

            yield audio, self.is_speech

            self.samples_count += samples
            self.audio_buffer = self.audio_buffer[samples:]
