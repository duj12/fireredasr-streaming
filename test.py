import asyncio
import argparse
import time
import os
import glob
import librosa
import numpy as np
import sounddevice as sd
from realtime_fireredasr import AsyncRealtimeSpeechRecognizer


async def run_single_stream(asr, wav_data, sid, chunk_size_s, results_log, realtime=False):
    """单个异步音频流任务"""
    start_time = time.time()
    first_token_time = None
    total_audio_sec = 0.0
    total_proc_time = 0.0

    chunk_samples = int(chunk_size_s * asr.sample_rate)
    chunk_bytes = chunk_samples * 2  # int16

    # === 模拟流式推送 ===
    for i in range(0, len(wav_data), chunk_samples):
        chunk = wav_data[i:i + chunk_samples]
        if len(chunk) == 0:
            break
        audio_bytes = (chunk * 32767).astype(np.int16).tobytes()

        t0 = time.time()
        results = await asr.recognize(sid, audio_bytes)
        t1 = time.time()

        proc_time = t1 - t0
        audio_time = len(audio_bytes) / (2 * asr.sample_rate)
        total_audio_sec += audio_time
        total_proc_time += proc_time

        for r in results:
            if r["type"] in ["begin", "changed", "end"]:
                print(f"[{sid[:6]}] {r}")
                if r["type"] == "begin" and first_token_time is None:
                    first_token_time = time.time() - start_time

        if realtime:
            await asyncio.sleep(chunk_size_s)  # 实时模式延时
        else:
            await asyncio.sleep(0)  # 快速并发模式

    # === 汇总性能 ===
    rtf = total_proc_time / total_audio_sec if total_audio_sec > 0 else 0
    latency = first_token_time or 0.0

    results_log.append({
        "sid": sid,
        "rtf": rtf,
        "latency": latency,
        "duration": total_audio_sec,
        "proc_time": total_proc_time,
    })

    print(f"\n[{sid[:6]}] DONE | Duration: {total_audio_sec:.2f}s | "
          f"RTF: {rtf:.3f} | 1stTok: {latency:.3f}s")


async def run_mic_stream(asr, sid, chunk_size_s, results_log):
    """麦克风实时识别流"""
    import queue
    import sounddevice as sd

    q = queue.Queue()
    sample_rate = asr.sample_rate

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=int(sample_rate * chunk_size_s),
        dtype="int16",
        channels=1,
        callback=callback,
    )

    print(f"[{sid[:6]}] 🎤 Start microphone input...")
    with stream:
        while True:
            data = q.get()
            if data == b"exit":
                break
            results = await asr.recognize(sid, data)
            for r in results:
                if r["type"] in ["begin", "changed", "end"]:
                    print(f"[MIC {sid[:6]}] {r}")


async def main(args):
    # === 初始化共享模型 ===
    asr = AsyncRealtimeSpeechRecognizer(
        model_dir=args.model_dir,
        use_gpu=not args.cpu,
        sample_rate=16000,
        silence_duration_s=args.vad_silence,
        transcribe_interval=args.interval,
    )

    print(f"✅ Model loaded ({args.model_dir})")

    # === 解析输入 ===
    wav_list = []
    if args.wavs.lower() == "none":
        wav_list = ["none"]
    else:
        # 支持逗号分隔
        wav_items = [w.strip() for w in args.wavs.split(",") if w.strip()]
        for item in wav_items:
            if os.path.isdir(item):
                wav_list.extend(glob.glob(os.path.join(item, "*.wav")))
                wav_list.extend(glob.glob(os.path.join(item, "*.flac")))
            elif os.path.isfile(item):
                wav_list.append(item)
            else:
                print(f"⚠️  跳过无效路径: {item}")

    if len(wav_list) == 0:
        print("❌ 未找到有效输入音频！")
        return

    # === 创建并发 session ===
    num_streams = args.num_concurrent if args.wavs.lower() != "none" else 1
    wav_list = (wav_list * num_streams)[:num_streams]
    sessions = [asr.create_session() for _ in range(num_streams)]
    results_log = []

    start_all = time.time()

    if args.wavs.lower() == "none":
        # 麦克风模式
        sid = asr.create_session()
        await run_mic_stream(asr, sid, args.chunk, results_log)
        return

    # === 预加载音频并重采样 ===
    all_wavs = []
    for path in wav_list:
        wav, sr = librosa.load(path, sr=None, mono=True)
        if sr != asr.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=asr.sample_rate)
        all_wavs.append(wav)

    print(f"🚀 Starting {num_streams} concurrent streams...")

    # === 并发执行 ===
    tasks = [
        run_single_stream(asr, wav, sid, args.chunk, results_log, realtime=args.realtime)
        for wav, sid in zip(all_wavs, sessions)
    ]
    await asyncio.gather(*tasks)

    total_time = time.time() - start_all
    total_audio = sum(r["duration"] for r in results_log)
    avg_rtf = np.mean([r["rtf"] for r in results_log]) if results_log else 0
    avg_latency = np.mean([r["latency"] for r in results_log]) if results_log else 0

    print("\n===== 🌍 Global Performance Summary =====")
    print(f"Total Sessions: {len(results_log)}")
    print(f"Total Audio Duration: {total_audio:.2f}s")
    print(f"Wall Time: {total_time:.2f}s")
    print(f"Overall RTF: {total_time / total_audio:.3f}")
    print(f"Avg Session RTF: {avg_rtf:.3f}")
    print(f"Avg 1stTok Latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async concurrent ASR stream tester")
    parser.add_argument("--model_dir", type=str, default="pretrained_models", help="Model path")
    parser.add_argument("--wavs", type=str, required=True,
                        help='Input wavs: file, comma list, folder, or "none" for mic')
    parser.add_argument("--num_concurrent", type=int, default=1, help="Number of concurrent streams")
    parser.add_argument("--vad_silence", type=float, default=0.25, help="Silence duration for VAD")
    parser.add_argument("--interval", type=float, default=0.5, help="Transcribe interval seconds")
    parser.add_argument("--chunk", type=float, default=0.5, help="Chunk size (seconds)")
    parser.add_argument("--realtime", action="store_true", help="Simulate real-time streaming")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    asyncio.run(main(args))
