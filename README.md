# FireRedAsr-Streaming

A low-latency realtime ASR based on [FireRedASR](https://github.com/FireRedTeam/FireRedASR) (a SOTA ASR model for Chinese and English speech recognizing)

## How it works

I've read the code of FireRedASR and found by utilizing the mechanism of autoregressive prediction in transformer decoder, we can input audio data into model in advance of the speeking sentence end.

## How to use

See `test.py`

``` python
from realtime_fireredasr import RealtimeSpeechRecognizer

# read wav from file
filename = "examples/wav/BAC009S0764W0121.wav"

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
```

Results (Tested on Tesla T4):
```
[{'type': 'begin', 'id': 0, 'text': None, 'ts': 0.672, 'latency': 0.0}]
[{'type': 'changed', 'id': 0, 'text': '甚至出现交', 'ts': 1.664, 'latency': 4.238659143447876}]
[{'type': 'changed', 'id': 0, 'text': '甚至出现交易几乎停', 'ts': 2.656, 'latency': 0.2656550407409668}]
[{'type': 'changed', 'id': 0, 'text': '甚至出现交易几乎停滞的情况', 'ts': 3.648, 'latency': 0.22229647636413574}]
[{'type': 'end', 'id': 0, 'text': '甚至出现交易几乎停滞的情况', 'ts': 4.128, 'latency': 0.11719179153442383}]
>>> wave EOF
```

Note that the recognition cost only 117ms at the sentence end, while it need nearly 400ms for whole sentence recognition without this project.
