"""Run only the Fig. 5 matrix; backend/model loading is deliberately lazy."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

from config_tuner.adapter import OnlineSemanticAdapter, select_similarity_threshold
from config_tuner.baselines import tune_baseline
from config_tuner.config_tuner import TwostageTuner
from config_tuner.knowledgebase import VideoKnowledgeBase
from config_tuner.latency_pruner import LatencyAwareSpacePruner, TokenLatencyModel
from config_tuner.objective import Objective
from config_tuner.records import TuningResult, canonical
from config_tuner.space import default_space
from experiments.backend import SentenceEmbedder, VideoBackend
from experiments.data import load_videos, split_queries
from util.runtime_config import DEFAULT_CONFIG, ROOT


def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding='utf-8'))
    if settings.get('schema_version') != 1:
        raise ValueError('Unsupported experiment configuration schema')
    runtime = settings['runtime']
    for name in ('data_root', 'preprocess_root', 'cache_root', 'output_root'):
        p = Path(runtime[name]).expanduser()
        runtime[name] = str(p if p.is_absolute() else ROOT / p)
    for model, profile in runtime['latency_profiles'].items():
        p = Path(profile).expanduser()
        runtime['latency_profiles'][model] = str(p if p.is_absolute() else ROOT / p)
    return settings


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def context_path(root, model, dataset, slo, seed):
    return Path(root) / model / dataset / f'slo_{slo:g}' / f'seed_{seed}'


def final_evaluation(result, backend, video, method, embedder=None, threshold=None):
    adapter = OnlineSemanticAdapter(video.texts, result, embedder, threshold) if method == 'Prysma' else None
    rows = []
    for question in video.questions:
        start = time.perf_counter()
        config = adapter.adapt(question['question']) if adapter else result.fallback
        adaptation_seconds = time.perf_counter() - start if adapter else 0.0
        observation = backend.evaluate(config, [question['question_id']])[0]
        # Cached extraction/inference retains its measured TTFT; online adaptation
        # is measured per request and is included in the final SLO decision.
        final_ttft = observation.ttft + adaptation_seconds
        rows.append({**asdict(observation), 'ttft': final_ttft, 'adapter_seconds': adaptation_seconds,
                     'config': asdict(config), 'quality': observation.raw_score if final_ttft <= result.history[0].slo else 0.0,
                     'tuning_query': question['question_id'] in result.train_ids})
    return rows


def run(settings, config_path, models=None, datasets=None, methods=None, seeds=None, slos=None):
    os.environ['PRYSMA_CONFIG'] = str(Path(config_path).resolve())
    runtime = settings['runtime']
    root = Path(runtime['output_root'])
    fingerprint = hashlib.sha256(canonical(settings).encode()).hexdigest()[:24]
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Output directory belongs to another experiment configuration')
    else:
        manifest = {'schema_version': 1, 'fingerprint': fingerprint, 'settings': settings, 'datasets': {}}
    selected_models = models or settings['models']
    selected_datasets = datasets or list(settings['datasets'])
    selected_methods = methods or settings['methods']
    selected_seeds = seeds or settings['seeds']
    selected_slos = slos or settings['slos']
    for chosen, allowed in ((selected_models, settings['models']), (selected_datasets, settings['datasets']),
                            (selected_methods, settings['methods']), (selected_seeds, settings['seeds']),
                            (selected_slos, settings['slos'])):
        if not set(chosen).issubset(allowed):
            raise ValueError('Requested experiment selection is outside the configured matrix')
    for dataset in selected_datasets:
        ds = settings['datasets'][dataset]
        videos = load_videos(runtime['data_root'], dataset, ds['video_limit'])
        if not videos:
            raise ValueError(f'No videos selected for {dataset}')
        dataset_manifest = {v.video_id: [q['question_id'] for q in v.questions] for v in videos}
        qa_digest = hashlib.sha256(canonical([v.questions for v in videos]).encode()).hexdigest()
        prior = manifest['datasets'].get(dataset)
        if prior and (prior['videos'] != dataset_manifest or prior['qa_digest'] != qa_digest):
            raise ValueError('Dataset annotations/selection changed; choose a new output directory')
        manifest['datasets'][dataset] = {'videos': dataset_manifest, 'qa_digest': qa_digest}
        atomic_json(manifest_path, manifest)
        space = default_space(dataset)
        for model in selected_models:
            embedder = SentenceEmbedder(runtime['sentence_model'], runtime['device']) if 'Prysma' in selected_methods else None
            features = None
            for slo, seed in itertools.product(selected_slos, selected_seeds):
                profile = None
                if 'Prysma' in selected_methods:
                    profile = TokenLatencyModel.load(runtime['latency_profiles'][model], model, runtime['serving_context'])
                context = {'model': model, 'dataset': dataset, 'slo': slo, 'seed': seed,
                           'fingerprint': fingerprint, 'qa_digest': qa_digest}
                prysma_context = {**context, 'profile_digest': hashlib.sha256(canonical(profile.to_dict()).encode()).hexdigest()} if profile else None
                folder = context_path(root, model, dataset, slo, seed)
                cache_fingerprint = hashlib.sha256(canonical(context).encode()).hexdigest()[:24]
                backends = {v.video_id: VideoBackend(runtime, model, dataset, v, cache_fingerprint) for v in videos}
                splits = {v.video_id: split_queries(v, ds['train_ratio'], seed)[0] for v in videos}
                kb = None
                if 'Prysma' in selected_methods:
                    kb_path = folder / 'knowledgebase.json'
                    if kb_path.is_file():
                        kb = VideoKnowledgeBase.load(kb_path, prysma_context)
                    else:
                        if features is None:
                            features = {v.video_id: backends[v.video_id].feature() for v in videos}
                        kb = VideoKnowledgeBase(prysma_context, features)
                        kb.build(settings['knowledge_base_fraction'], seed)
                        kb.save(kb_path)
                    for vid in kb.video_ids:
                        if vid not in kb.results:
                            pruner = LatencyAwareSpacePruner(slo, profile)
                            objective = Objective(backends[vid], splits[vid], slo, pruner)
                            kb.add_result(vid, TwostageTuner(space, seed).tune(objective, settings['budget']))
                            kb.save(kb_path)
                    if kb.similarity_threshold is None:
                        kb.similarity_threshold = select_similarity_threshold(kb.results,
                            {v.video_id: v.texts for v in videos}, embedder, settings['similarity_thresholds'])
                        kb.save(kb_path)
                for method, video in itertools.product(selected_methods, videos):
                    method_context = prysma_context if method == 'Prysma' else context
                    destination = folder / method / 'videos' / (video.video_id + '.json')
                    if destination.is_file():
                        previous = json.loads(destination.read_text(encoding='utf-8'))
                        if previous['context'] != method_context or previous['method'] != method:
                            raise ValueError('Existing result context mismatch')
                        if previous.get('complete'):
                            continue
                    train_ids = splits[video.video_id]
                    history_path = folder / method / 'histories' / (video.video_id + '.json')
                    if history_path.is_file():
                        saved = json.loads(history_path.read_text(encoding='utf-8'))
                        if saved['context'] != method_context or saved['method'] != method:
                            raise ValueError('Existing tuning history context mismatch')
                        result = TuningResult.from_dict(saved['tuning'])
                    elif method == 'Prysma' and video.video_id in kb.results:
                        result = kb.results[video.video_id]
                    else:
                        pruner = LatencyAwareSpacePruner(slo, profile) if method == 'Prysma' else None
                        objective = Objective(backends[video.video_id], train_ids, slo, pruner)
                        if method == 'Prysma':
                            tuner = TwostageTuner(space, seed, kb, video.video_id)
                            result = tuner.tune(objective, settings['budget'])
                        else:
                            result = tune_baseline(method, objective, space, settings['budget'], seed)
                    if result.train_ids != train_ids:
                        raise ValueError('Tuning history split does not match the paired experiment split')
                    atomic_json(history_path, {'context': method_context, 'method': method, 'tuning': result.to_dict()})
                    predictions = final_evaluation(result, backends[video.video_id], video, method, embedder,
                                                   kb.similarity_threshold if kb else None)
                    atomic_json(destination, {'schema_version': 1, 'complete': True, 'context': method_context, 'method': method,
                                             'video_id': video.video_id, 'train_ids': train_ids,
                                             'budget_used': result.budget_used, 'predictions': predictions})
                    print(f'{model} {dataset} SLO={slo:g} seed={seed} {method} video={video.video_id}: saved', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate main-paper Fig. 5 experiment records')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model', nargs='+')
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--method', nargs='+')
    parser.add_argument('--seed', nargs='+', type=int)
    parser.add_argument('--slo', nargs='+', type=float)
    parser.add_argument('--list', action='store_true', help='List matrix without loading data, models or services')
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.list:
        dimensions = [args.model or settings['models'], args.dataset or settings['datasets'],
                      args.slo or settings['slos'], args.seed or settings['seeds'], args.method or settings['methods']]
        for model, dataset, slo, seed, method in itertools.product(*dimensions):
            print(f'{model}\t{dataset}\t{slo:g}\t{seed}\t{method}')
        return
    run(settings, args.config, args.model, args.dataset, args.method, args.seed, args.slo)


if __name__ == '__main__':
    main()
