import os
import numpy as np
import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
import functools


@functools.lru_cache
def load_model(model_dir):
    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    feat_extractor = ASRFeatExtractor(cmvn_path)

    model_path = os.path.join(model_dir, "model.pth.tar")
    package = torch.load(model_path, map_location=lambda storage,
                         loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()

    dict_path = os.path.join(model_dir, "dict.txt")
    spm_model = os.path.join(model_dir, "train_bpe1000.model")
    tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)

    return feat_extractor, model, tokenizer


class FireRedAsrStreaming:
    def __init__(self,
                 model_dir="pretrained_models",
                 use_gpu=False,
                 sample_rate=16000,
                 least_ys_state_len=4):
        self.use_gpu = use_gpu
        self.sample_rate = sample_rate
        self.least_ys_state_len = least_ys_state_len
        feat_extractor, model, tokenizer = load_model(model_dir)
        self.feat_extractor = feat_extractor
        self.model = model
        if use_gpu:
            self.model = model.cuda()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, wav_buffer, asr_state, full_update=False, args={}):
        feat = self.feat_extractor(self.sample_rate, wav_buffer)
        feats = feat.unsqueeze(0)
        lengths = torch.tensor([feat.size(0)]).long()
        if self.use_gpu:
            feats, lengths = feats.cuda(), lengths.cuda()

        hyps = self.model.transcribe(
            feats, lengths,
            args.get("beam_size", 1),
            args.get("nbest", 1),
            args.get("decode_max_len", 0),
            args.get("softmax_smoothing", 1.0),
            args.get("aed_length_penalty", 0.0),
            args.get("eos_penalty", 1.0),
            None if full_update else asr_state,
        )

        hyp = hyps[0][0]
        ys = hyp["yseq"]
        hyp_ids = [int(id) for id in ys.cpu()]
        text = self.tokenizer.detokenize(hyp_ids)

        if len(ys) > self.least_ys_state_len:
            asr_state = ys[:-1] if len(ys) > 0 else ys

        return text, asr_state
