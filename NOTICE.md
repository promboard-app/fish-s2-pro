# NOTICE

These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.

Built with Fish Audio.

## Fish Audio Materials in this plugin

- `nodes/fish_speech/` — the `fish_speech` package from [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (main, commit 541566932abb), under the Fish Audio Community License v1.0 (`LICENSE-fish-audio.md`).

## Changes made by PromBoard (Derivative Work)

PromBoard changed the Fish Audio Materials so that they install from wheels only (no source builds), as follows. Each changed file also says so in its header.

- Left out: `fish_speech/inference_engine/`, `fish_speech/models/dac/inference.py`, `fish_speech/utils/schema.py`, `fish_speech/utils/file.py` (API server and command-line pieces this node does not use).
- `fish_speech/models/text2semantic/inference.py`: `load_codec_model` builds the codec with `fish_speech/utils/instantiate.py` instead of `hydra.utils.instantiate(OmegaConf.load(...))` (two imports and two lines changed).
- `fish_speech/models/dac/modded_dac.py`: removed the unused `audiotools.AudioSignal` import and uses `torch.nn.Module` instead of `audiotools.ml.BaseModel`.
- Added `fish_speech/utils/instantiate.py` (written by PromBoard, Apache-2.0): the part of hydra's `instantiate` that `configs/modded_dac_vq.yaml` uses (`_target_`, `_partial_`, YAML anchors).

The codec built this way has the same 535 parameter names and shapes as the official `codec.pth`.

## Other parts

- `nodes/dac/` — three files from descript-audio-codec 1.0.0 (MIT, Copyright (c) 2023-present, Descript; `nodes/dac/LICENSE`). `model/base.py` imports `audiotools` only if it is installed (header comment). The `__init__.py` files are PromBoard's.
- `nodes/main.py` and the rest of this repository — PromBoard, Apache-2.0 (`LICENSE`).
- The model weights are not in this repository. The app downloads them from the official Hugging Face repository under that model's own license.
