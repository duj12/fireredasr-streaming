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


# read wav from file
filename = "examples/wav/BAC009S0764W0121.wav"

# capture wav from microphone
# filename = None

# create realtime speech recognizer instance
asr = RealtimeSpeechRecognizer(
    model_dir="pretrained_models",  # models' dir
    use_gpu=False,  # use gpu or not
    sample_rate=16000,  # audio sample rate
    silence_duration_s=0.4,  # silence duration for VAD cutting
    transcribe_interval=1.0,  # how many seconds to transcribe once
)

# recognization loop
stream = open_stream(filename)
while True:
    data = stream.read(1600)
    if data == b"":
        print(">>>wave EOF")
        break
    results = asr.recognize(data)
    if len(results) > 0:
        print(results)
