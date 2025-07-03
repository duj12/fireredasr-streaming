import pyaudio
from realtime_fireredasr import RealtimeSpeechRecognizer


def open_stream(filename=None):
    stream = None
    if filename is None:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1600)
    else:
        stream = open(filename, "rb")
    return stream


filename = "examples/wav/BAC009S0764W0121.wav"
# filename = None

model_dir = "pretrained models"
asr = RealtimeSpeechRecognizer()

stream = open_stream(filename)
while True:
    data = stream.read(1600)
    if data == b"":
        print(">>>wave EOF")
        break
    results = asr.recognize(data)
    if len(results) > 0:
        print(results)
