import time
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
    use_gpu=True,  # use gpu or not
    sample_rate=16000,  # audio sample rate
    silence_duration_s=0.4,  # silence duration for VAD cutting
    transcribe_interval=0.5,  # how many seconds to transcribe once
)

print('Load model finished, start streaming recognize...')
start_time = None             
first_token_time = None       
total_audio_sec = 0.0       
total_proc_time = 0.0 
# recognization loop
stream = open_stream(filename)
while True:
    data = stream.read(1600)
    if data == b"":
        print(">>>wave EOF")
        break
    if start_time is None:
        start_time = time.time()

    t0 = time.time()
    results = asr.recognize(data)
    t1 = time.time()
    
    proc_time = t1 - t0
    audio_time = len(data) / (2 * 16000)  # int16, 2byte per sample    
    total_proc_time += proc_time
    total_audio_sec += audio_time
    
    if len(results) > 0:
        print(results)
        text = results[-1]['text']
        if text is not None:
            if first_token_time is None:
                first_token_time = time.time() - start_time

if total_audio_sec > 0:
    rtf = total_proc_time / total_audio_sec
    print("\n===== Summary =====")
    if first_token_time:
        print(f"1stTok Latency: {first_token_time:.3f} s")
    print(f"Total Duration: {total_audio_sec:.2f} s")
    print(f"ASR Processing: {total_proc_time:.2f} s")
    print(f"RealTimeFactor: {rtf:.3f}")