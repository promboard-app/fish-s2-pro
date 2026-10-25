# PromBoard 가 더한 파일(Apache-2.0) — 원본 fish-speech 에는 없다.
# 원본 load_codec_model 은 hydra.utils.instantiate(OmegaConf.load(...)) 로 configs/modded_dac_vq.yaml 을 객체로 만든다.
# hydra-core · omegaconf 는 antlr4-python3-runtime(휠 없음)을 끌고 와 PromBoard 의 휠 전용 설치가 안 되므로, 이 설정 파일이 쓰는
# 기능(_target_ · _partial_ · YAML 앵커)만 같은 뜻으로 옮겼다. 숫자 읽기는 OmegaConf 와 같게 1e-5 같은 표기도 실수로 읽는다.
import functools
import importlib
import re

import yaml


class _Loader(yaml.SafeLoader):
    pass


# OmegaConf 의 YAML 로더와 같은 실수 규칙(YAML 1.1 기본 규칙은 점 없는 1e-5 를 글자로 읽는다)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def _locate(target):
    module, _, name = target.rpartition(".")
    return getattr(importlib.import_module(module), name)


def instantiate(node):
    """hydra.utils.instantiate 의 이 설정에 쓰는 부분: 안쪽부터 _target_ 을 만들고, _partial_ 이면 functools.partial 로 둔다."""
    if isinstance(node, list):
        return [instantiate(v) for v in node]
    if not isinstance(node, dict):
        return node
    kwargs = {k: instantiate(v) for k, v in node.items() if k not in ("_target_", "_partial_")}
    if "_target_" not in node:
        return kwargs
    target = _locate(node["_target_"])
    if node.get("_partial_"):
        return functools.partial(target, **kwargs)
    return target(**kwargs)
