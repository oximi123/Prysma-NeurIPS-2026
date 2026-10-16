"""Collect or import real measurements; build the Appendix G calibration profile.

No default latency coefficients or synthetic measurement files are supplied.
"""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
from statistics import mean
import numpy as np
from config_tuner.latency_pruner import DEFAULT_MONOTONE, KnobSpec, TokenLatencyModel
from config_tuner.records import Configuration, canonical
from config_tuner.space import default_space
from experiments.backend import VideoBackend
from experiments.data import load_videos
from experiments.run_main_result import atomic_json, load_settings
from util.runtime_config import DEFAULT_CONFIG


def build_profile(measurements, model, serving_context, space):
    counts, overheads, ratios = {}, {}, []
    rows = list(measurements)
    for row in rows:
        if row.get('model') != model or row.get('serving_context') != serving_context:
            raise ValueError('Measurement model or serving conditions do not match')
        config = Configuration.from_dict(row['config'])
        n, ttft = int(row['token_count']), float(row['timing']['llm_ttft'])
        if n <= 0 or not np.isfinite(ttft) or ttft < 0:
            raise ValueError('Invalid measured token count/TTFT')
        counts.setdefault(n, []).append(ttft)
        overhead = float(row['ttft']) - ttft
        if overhead < -1e-10 or not np.isfinite(overhead):
            raise ValueError('Invalid measured end-to-end overhead')
        key = '+'.join(config.mc) + '|' + config.retrieval_model
        overheads.setdefault(key, []).append(max(0., overhead))
        frames = config.modalities.get('visiontoken', {}).get('num_frame')
        tokens = row.get('modality_tokens', {}).get('visiontoken')
        if frames and tokens:
            ratios.append(float(tokens) / frames)
    if len(counts) < 2 or not ratios:
        raise ValueError('Calibration requires multiple prompt lengths and measured VT token counts')
    # Each MC/retriever needs an overhead entry. No zero fallback for unmeasured combinations.
    expected = {'+'.join(mc) + '|' + retrieval for mc in space.combinations()
                for retrieval in (space.retrieval_models if set(mc) & {'object', 'ocr', 'asr'} else ['None'])}
    missing = expected - set(overheads)
    if missing:
        raise ValueError(f'Missing measured MC/retriever overheads: {sorted(missing)}')
    monotone = []
    for spec in DEFAULT_MONOTONE:
        if spec.modality not in space.modalities:
            continue
        sweeps = {}
        expected_values = set(space.modalities[spec.modality][spec.name])
        for row in rows:
            config = Configuration.from_dict(copy.deepcopy(row['config']))
            if spec.modality not in config.modalities:
                continue
            value = config.modalities[spec.modality].pop(spec.name)
            group = (row['video_id'], row['question_id'], canonical(vars(config)))
            sweeps.setdefault(group, {}).setdefault(value, []).append(row['token_count'])
        full = [v for v in sweeps.values() if set(v) == expected_values]
        if not full:
            continue
        sequences = [[mean(s[v]) for v in sorted(expected_values)] for s in full]
        if all(all(b >= a for a, b in zip(s, s[1:])) for s in sequences):
            monotone.append(KnobSpec(spec.modality, spec.name, 'increasing'))
        elif all(all(b <= a for a, b in zip(s, s[1:])) for s in sequences):
            monotone.append(KnobSpec(spec.modality, spec.name, 'decreasing'))
    tokens = sorted(counts)
    ttfts = np.maximum.accumulate([mean(counts[n]) for n in tokens]).tolist()
    return TokenLatencyModel(model, serving_context, tokens, ttfts, min(ratios),
                             {key: mean(values) for key, values in overheads.items()}, monotone)


def collect(settings, model, dataset, destination, video_count=3, repeats=3):
    runtime = settings['runtime']
    space = default_space(dataset)
    videos = load_videos(runtime['data_root'], dataset, video_count)
    if not videos:
        raise ValueError('No calibration videos')
    rng = np.random.default_rng(42)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as output:
        for video in videos:
            backend = VideoBackend(runtime, model, dataset, video, 'latency_calibration')
            ids = [video.questions[0]['question_id']]
            candidates = []
            for mc in space.combinations():
                retrievals = space.retrieval_models if set(mc) & {'object', 'ocr', 'asr'} else ['None']
                for retrieval in retrievals:
                    base = space.random(rng, mc)
                    for m, knobs in base.modalities.items():
                        for k in knobs:
                            knobs[k] = space.modalities[m][k][0]
                    base.retrieval_model, base.retrieval_threshold = retrieval, .3
                    candidates.append(space.normalize(base))
            # Controlled sweeps: every other knob, video and question stays fixed.
            for spec in DEFAULT_MONOTONE:
                if spec.modality not in space.modalities:
                    continue
                base = space.random(rng, (spec.modality,))
                for value in space.modalities[spec.modality][spec.name]:
                    candidate = copy.deepcopy(base)
                    candidate.modalities[spec.modality][spec.name] = value
                    candidates.append(candidate)
            for candidate in candidates:
                for _ in range(repeats):
                    for row in backend.measure(candidate, ids):
                        row.update(model=model, serving_context=runtime['serving_context'])
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                print(f'Calibrated {video.video_id} {candidate.mc}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='MSRVTT-QA', help='Use an audio-bearing dataset to profile all five modalities')
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--collect', action='store_true', help='Run extraction/inference to collect real calibration samples')
    parser.add_argument('--videos', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    os.environ['PRYSMA_CONFIG'] = str(args.config.resolve())
    settings = load_settings(args.config)
    if args.collect:
        collect(settings, args.model, args.dataset, args.measurements, args.videos, args.repeats)
    rows = [json.loads(line) for line in args.measurements.read_text(encoding='utf-8').splitlines() if line.strip()]
    profile = build_profile(rows, args.model, settings['runtime']['serving_context'], default_space(args.dataset))
    destination = settings['runtime']['latency_profiles'][args.model]
    atomic_json(destination, profile.to_dict())
    print(f'Saved measured profile to {destination}')


if __name__ == '__main__':
    main()
