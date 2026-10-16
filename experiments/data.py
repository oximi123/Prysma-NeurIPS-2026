"""Annotation loading and paired, deterministic per-video tuning splits."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


@dataclass
class Video:
    video_id: str
    path: Path
    questions: list[dict]

    @property
    def texts(self):
        return {q['question_id']: q['question'] for q in self.questions}


def load_videos(data_root, dataset, limit=None):
    folder = Path(data_root) / dataset
    if dataset == 'EgoTimeQA' and not folder.exists():
        folder = Path(data_root) / 'GroundQA'
    file_groups = {
        'MSVD-QA': [('train.jsonl', 'val.jsonl', 'test.jsonl'), ('train_qa.json', 'val_qa.json', 'test_qa.json')],
        'MSRVTT-QA': [('train_qa.json', 'val_qa.json', 'test_qa.json')],
        'EgoTimeQA': [('annotations.EgoTimeQA.json',)],
    }
    if dataset not in file_groups:
        raise ValueError(f'Unsupported main-paper dataset: {dataset}')
    files = next((names for names in file_groups[dataset] if all((folder / n).is_file() for n in names)), None)
    if files is None:
        raise FileNotFoundError(f'Annotation files missing in {folder}; expected one of {file_groups[dataset]}')
    questions = {}
    for name in files:
        text = (folder / name).read_text(encoding='utf-8')
        entries = [json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(text)
        if not isinstance(entries, list):
            raise ValueError(f'{name} must contain a list of QA records')
        for entry in entries:
            vid = str(entry['video_id'])
            records = questions.setdefault(vid, [])
            records.append({**entry, 'video_id': vid, 'question_id': f'{vid}_{len(records)}',
                            'answer': entry.get('answer', entry.get('a'))})
            if records[-1]['answer'] is None or not isinstance(records[-1]['question'], str):
                raise ValueError(f'Missing reference answer/question for {vid}')
    paths = {}
    for path in sorted((folder / 'videos').iterdir()):
        if path.suffix.lower() not in ('.mp4', '.avi', '.mkv', '.webm'):
            continue
        vid = path.stem
        if dataset == 'MSRVTT-QA' and vid.startswith('video'):
            vid = vid[5:]
        if vid in paths:
            raise ValueError(f'Ambiguous video ID {vid}')
        paths[vid] = path
    missing = set(questions) - set(paths)
    if missing:
        raise FileNotFoundError(f'{len(missing)} annotated videos are missing, e.g. {sorted(missing)[:3]}')
    ids = sorted(questions, key=lambda v: (-len(questions[v]), v))
    if limit is not None:
        ids = ids[:limit]
    return [Video(v, paths[v], questions[v]) for v in ids]


def split_queries(video, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError('Tuning fraction must lie strictly between 0 and 1')
    ids = [q['question_id'] for q in video.questions]
    if len(ids) < 2:
        raise ValueError('Each video needs at least two queries for tuning/evaluation separation')
    salt = int.from_bytes(hashlib.sha256(video.video_id.encode()).digest()[:8], 'little')
    chosen = set(random.Random(seed ^ salt).sample(ids, min(len(ids) - 1, max(1, int(len(ids) * ratio)))))
    return [q for q in ids if q in chosen], [q for q in ids if q not in chosen]
