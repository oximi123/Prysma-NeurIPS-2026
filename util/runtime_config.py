"""Shared paths/checkpoints for the experiment process and extraction servers."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'main_result.json'


def configuration():
    path = Path(os.environ.get('PRYSMA_CONFIG', str(DEFAULT_CONFIG)))
    data = json.loads(path.read_text(encoding='utf-8'))
    runtime = data['runtime'].copy()
    for name in ('data_root', 'preprocess_root', 'cache_root', 'output_root'):
        value = Path(runtime[name]).expanduser()
        runtime[name] = str(value if value.is_absolute() else ROOT / value)
    return runtime


def model_path(name):
    aliases = {'LlavaVideoQwen7b': 'LLaVA-Video-7B', 'VideoLlamaMistral7b': 'VideoLLaMA2-7B'}
    runtime = configuration()
    value = runtime['model_paths'][aliases.get(name, name)]
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def dataset_path(name):
    root = Path(configuration()['data_root'])
    canonical = 'EgoTimeQA' if name == 'GroundQA' else name
    path = root / canonical
    if canonical == 'EgoTimeQA' and not path.exists():
        path = root / 'GroundQA'
    return str(path)
