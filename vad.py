import functools
import numpy as np


@functools.lru_cache
def get_vad_model(model_dir="pretrained_models"):
    # now is silero_vad v5 model
    return SileroVADModel(f"{model_dir}/silero_vad.onnx")


class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def get_initial_state(self, batch_size: int):
        return np.zeros((2, batch_size, 128), dtype=np.float32)


def __call__(self, x, state, sr: int):
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    if len(x.shape) > 2:
        raise ValueError(
            f"Too many dimensions for input audio chunk {len(x.shape)}"
        )
    if sr/x.shape[1] > 31.25:
        raise ValueError("Input audio chunk is too short")

    ort_inputs = {
        "input": x,
        "state": state,
        "sr": np.array(sr, dtype="int64"),
    }

    out, state = self.session.run(None, ort_inputs)

    return out, state
