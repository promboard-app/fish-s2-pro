"""promboard.fish-s2-pro — Fish Audio S2 Pro 목소리 노드 코드.

노드 하나(fish-s2-pro)가 글을 읽어 목소리 WAV(44.1kHz 모노)를 낸다. 참조 목소리(+ 그 대사)를 이으면 그 목소리로 복제하고,
글 안에 [whisper] · [laughing] · [excited] 같은 자유 태그로 감정 · 말투를 바꾼다. 화자를 나눠 쓰면(Speaker 1: … / Speaker 2: …)
화자마다 참조 목소리를 따로 이을 수 있다(대본 화자 최대 4명, 참조 목소리는 화자 1~3).

공식 코드: fishaudio/fish-speech(main 541566932abb, 2026-10-14)의 fish_speech 패키지를 이 폴더 nodes/fish_speech 에 그대로 담았다
  (Fish Audio Community License v1.0 — LICENSE-fish-audio.md · NOTICE.md). 이 노드는 그 안의 text2semantic.inference(init_model ·
  load_codec_model · generate_long · decode_to_audio)를 서버(tools/server)와 같은 순서로 부른다. 샘플링 기본값은 공식 SDK 값(sampling.py)이다.
  고친 곳(휠만으로 설치되게): hydra-core · omegaconf 대신 fish_speech/utils/instantiate.py, descript-audiotools 를 빼고
  descript-audio-codec 1.0.0 파일 3개(MIT)를 nodes/dac 에 담았다 — 고친 파일마다 머리 주석, 목록은 NOTICE.md.
대본: 'Speaker 1: 글'(1~4) 또는 공식 표기 <|speaker:0|>. 화자 표시가 없으면 화자 1 하나로 읽는다. 공식 generate_long 은 화자 표시
  단위로 나눠(한 묶음 chunk_length 바이트) 차례로 만들고, 앞 묶음의 소리를 다음 묶음의 문맥으로 이어 목소리를 유지한다.
문맥 길이(max_seq_len): 공식 설정은 32,768 토큰이고 그만큼 KV 캐시를 미리 잡는다(bf16 약 4.8GB). 그래픽카드 메모리가 모자라면 줄인다.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 받은 파일만 쓴다(실행 환경은 인터넷이 막혀 있다)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # 담아 온 fish_speech 패키지를 그 이름 그대로 불러온다(공식 코드가 절대 경로로 import 한다)

MODEL_FILE = "fishaudio/s2-pro/config.json"
MAX_REF_VOICES = 3  # 참조 목소리를 받는 화자 수(노드 입력 칸 상한 8 — 글 1 + 소리 3 + 대사 3)
MAX_TEXT_CHARS = 4000  # 문맥 8,192 토큰(약 4분 40초 소리)에 들어가고 앱이 노드 한 번에 주는 시간 안에 끝나게
MAX_REF_SEC = 60.0
SPEAKER_RE = re.compile(r"^\s*(?:speaker\s*)?([1-4])\s*[:：]\s*(.*)$", re.IGNORECASE)
OFFICIAL_TAG_RE = re.compile(r"<\|speaker:\d+\|>")

_STATE = {}


def _load(ctx, max_seq_len):
    import torch

    device = ctx.device
    if _STATE.get("device") == device:
        model = _STATE["model"]
        if model.config.max_seq_len != max_seq_len:  # 문맥 길이만 바뀌면 다시 올리지 않고 KV 캐시만 새로 잡게 한다
            model.config.max_seq_len = max_seq_len
            model._cache_setup_done = False
        return model, _STATE["decode"], _STATE["codec"]
    from fish_speech.models.text2semantic.inference import init_model, load_codec_model

    ctx.progress(0.02, "load-model")  # 올리는 동안 파이썬이 오래 멈춘다 — 바로 앞에 알린다
    start = time.monotonic()
    _STATE.clear()
    if device == "cuda":
        torch.cuda.empty_cache()
    folder = os.path.dirname(ctx.model_path(MODEL_FILE))
    precision = torch.float32 if device == "cpu" else torch.bfloat16  # 서버 기본(--half 아님)은 bfloat16
    try:
        model, decode = init_model(folder, device, precision, compile=False)
        model.config.max_seq_len = max_seq_len  # KV 캐시 크기(generate 가 이 값으로 setup_caches 한다)
        codec = load_codec_model(os.path.join(folder, "codec.pth"), device, precision)
    except torch.cuda.OutOfMemoryError:
        _STATE.clear()
        torch.cuda.empty_cache()
        raise RuntimeError(
            "그래픽카드 메모리가 모자라 모델을 올리지 못했습니다(가중치만 약 10GB). 다른 AI 작업 · 프로그램을 닫고 다시 해 주세요."
        ) from None
    _STATE.update(device=device, model=model, decode=decode, codec=codec)
    ctx.log(f"S2 Pro + codec loaded on {device} ({precision}) in {time.monotonic() - start:.1f}s, max_seq_len {max_seq_len}")
    return model, decode, codec


def _read_audio(path, rate):
    """참조 목소리 → float32 모노 텐서 (T,) at rate. wav · flac · ogg · mp3 는 soundfile, 그 밖은 PyAV."""
    import numpy as np
    import torch
    import torchaudio

    try:
        import soundfile as sf

        d, sr = sf.read(path, dtype="float32", always_2d=True)
        data = d.mean(axis=1)
    except Exception:
        import av

        try:
            with av.open(path) as container:
                if not container.streams.audio:
                    raise ValueError("참조 목소리 파일에 소리가 없습니다.")
                resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
                chunks = [o.to_ndarray().reshape(-1) for fr in container.decode(audio=0) for o in resampler.resample(fr)]
                chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
        except av.error.FFmpegError as exc:
            raise ValueError(f"참조 목소리 파일을 읽지 못했습니다. wav · mp3 · flac · ogg · m4a 소리인지 확인해 주세요. ({exc})") from None
        data, sr = (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), rate
    seconds = data.size / sr
    if seconds < 1.0:
        raise ValueError("참조 목소리가 너무 짧습니다. 10~30초 길이의 말소리가 가장 좋습니다.")
    if seconds > MAX_REF_SEC:
        raise ValueError(f"참조 목소리가 너무 깁니다({seconds:.0f}초). {MAX_REF_SEC:.0f}초까지 받습니다.")
    wav = torch.from_numpy(data.astype(np.float32))
    if sr != rate:
        wav = torchaudio.functional.resample(wav, sr, rate)
    return wav


def to_official_script(text):
    """'Speaker N: 글' 줄 → 공식 <|speaker:N-1|>글. 공식 표기가 이미 있으면 그대로. 화자 표시가 전혀 없으면 문단마다 <|speaker:0|>.
    (공식 generate_long 은 화자 표시 단위로만 나눠 묶는다 — 표시가 없으면 글 전체를 한 번에 만들다 길이 상한에 걸린다.)"""
    if OFFICIAL_TAG_RE.search(text):
        return text
    out, speaker = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            speaker = int(m.group(1)) - 1
            line = m.group(2).strip()
            if not line:
                continue
        out.append(f"<|speaker:{0 if speaker is None else speaker}|>{line}")
    return "\n".join(out)


def fish_s2_pro(ctx, inputs, settings):
    import numpy as np
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.errors import InvalidInputError
    from fish_speech.models.text2semantic.inference import decode_to_audio, generate_long

    text = (inputs.get("text") or "").strip()
    if not text:
        raise ValueError("읽을 글을 이어 주세요.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"글이 너무 깁니다({len(text)}자). 한 번에 {MAX_TEXT_CHARS}자까지 읽습니다 — 나눠서 이어 주세요.")
    script = to_official_script(text)

    refs = []
    for n in range(1, MAX_REF_VOICES + 1):
        path = inputs.get(f"voice-{n}")
        if not path:
            continue
        ref_text = (inputs.get(f"voice-{n}-text") or "").strip()
        if not ref_text:
            raise ValueError(f"화자 {n} 참조 목소리의 대사를 '화자 {n} 참조 대사'에 이어 주세요(참조 소리와 그 대사를 같이 받습니다).")
        refs.append((n - 1, path, ref_text))

    model, decode, codec = _load(ctx, int(settings["context"]))
    device = ctx.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    prompt_tokens, prompt_texts = [], []
    for speaker, path, ref_text in refs:
        ctx.check_cancel()
        ctx.progress(0.08, "prepare")
        wav = _read_audio(path, codec.sample_rate).to(device)
        dtype = next(codec.parameters()).dtype
        with torch.inference_mode():  # 공식 encode_audio 와 같은 순서(파일 대신 읽어 둔 소리)
            indices, lengths = codec.encode(wav[None, None].to(dtype), torch.tensor([wav.shape[0]], device=device, dtype=torch.long))
        prompt_tokens.append(indices[0, :, : lengths[0]])
        prompt_texts.append(f"<|speaker:{speaker}|>{ref_text}")

    torch.manual_seed(int(settings["seed"]))
    batches_total = max(1, len(re.findall(r"<\|speaker:\d+\|>", script)))
    pieces = []
    start = time.monotonic()
    try:
        made = 0
        for response in generate_long(
            model=model,
            device=device,
            decode_one_token=decode,
            text=script,
            max_new_tokens=int(settings["max-tokens"]),
            top_p=float(settings["top-p"]),
            temperature=float(settings["temperature"]),
            compile=False,
            iterative_prompt=True,
            chunk_length=int(settings["chunk"]),
            prompt_text=prompt_texts or None,
            prompt_tokens=prompt_tokens or None,
        ):
            ctx.check_cancel()
            if response.action != "sample":
                break
            with torch.inference_mode():
                audio = decode_to_audio(response.codes.to(device), codec)
            pieces.append(audio.float().cpu().numpy())
            made += max(1, len(re.findall(r"<\|speaker:\d+\|>", response.text or "")))
            ctx.progress(0.1 + 0.85 * min(1.0, made / batches_total), "generate")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "그래픽카드 메모리가 모자랍니다. 설정 '문맥 길이'를 줄이거나 다른 AI 작업 · 프로그램을 닫고 다시 해 주세요."
        ) from None
    except InvalidInputError as exc:
        if "too long" in str(exc):
            raise ValueError(
                "글과 참조 목소리가 문맥 길이를 넘었습니다. 설정 '문맥 길이'를 늘리거나 글 · 참조 목소리를 짧게 나눠 주세요."
            ) from None
        raise
    took = time.monotonic() - start
    if not pieces:
        raise RuntimeError("목소리가 만들어지지 않았습니다. 글을 바꿔 다시 해 주세요.")
    wav = np.concatenate(pieces).astype(np.float32)
    out = ctx.out_path("fish-s2-pro.wav")
    sf.write(out, wav, codec.sample_rate, subtype="PCM_16")
    peak = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0
    ctx.log(f"{len(pieces)} batches, {len(refs)} voices → {wav.size / codec.sample_rate:.1f}s audio in {took:.1f}s, peak VRAM reserved {peak:.0f} MiB")
    ctx.progress(1.0, "done")
    return {"audio": out}


NODES = {"fish-s2-pro": fish_s2_pro}
