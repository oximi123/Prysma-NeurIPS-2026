"""CPU-only checks; deterministic fixtures are never experiment result files."""
from __future__ import annotations
import ast
import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from config_tuner.adapter import OnlineSemanticAdapter, select_similarity_threshold
from config_tuner.baselines import OPPerTuneSearch, SCOOTSearch, tune_baseline
from config_tuner.config_tuner import EMMRMonitor, TPESearch, TwostageTuner, allocate_budget, estimate_mc_ei
from config_tuner.knowledgebase import VideoKnowledgeBase, similarity_weight, video_feature
from config_tuner.latency_pruner import KnobSpec, LatencyAwareSpacePruner, TokenLatencyModel, profile_monotonic_knobs
from config_tuner.objective import Objective
from config_tuner.records import Configuration, Observation, TrialRecord, TuningResult
from config_tuner.space import SearchSpace, default_space
from experiments.data import Video, split_queries
from experiments.backend import VideoBackend
from experiments.export_main_result import export_results
from experiments.profile_latency import build_profile
from experiments.run_main_result import ROOT, atomic_json, context_path, final_evaluation, load_settings, run


def cfg(frames=4, sampling='uniform'):
    return Configuration({'visiontoken': {'num_frame': frames, 'sampling_method': sampling}}, 'None', 0.)


def trial(config, scores, times=None, stage='video', cost=1):
    return TrialRecord(config, [Observation(q, 'fixture answer', s, (times or {}).get(q, .1))
                                for q, s in scores.items()], stage, cost, 1.)


def tiny_space():
    return SearchSpace({'visiontoken': {'num_frame': [4, 8, 16], 'sampling_method': ['uniform', 'keyframe']}},
                       ['None'], [0.])


def profile(space=None):
    space = space or tiny_space()
    overheads = {'+'.join(mc) + '|' + r: .01 for mc in space.combinations()
                 for r in (space.retrieval_models if set(mc) & {'asr', 'ocr', 'object'} else ['None'])}
    return TokenLatencyModel('fixture-model', {'batch_size': 1}, [0, 100, 1000], [0., .1, 2.], 10., overheads,
                             [KnobSpec('visiontoken', 'num_frame', 'increasing')])


class FixtureBackend:
    def __init__(self, *args):
        self.calls = []
        self.video = args[3] if args else None

    def feature(self):
        return np.asarray([1., .1 if self.video.video_id == 'a' else .2])

    def evaluate(self, config, ids):
        self.calls.append(list(ids))
        frame = config.modalities.get('visiontoken', {}).get('num_frame', 4)
        return [Observation(q, 'fixture answer', 5. if q.endswith('_0') or frame >= 8 else 2., .1,
                            40, {'visiontoken': 40} if 'visiontoken' in config.modalities else {}) for q in ids]


class FixtureTPE:
    def __init__(self, space, seed):
        self.space, self.rng = space, np.random.default_rng(seed)
        self.queue = []

    def enqueue(self, config, mc):
        self.queue.append(config)

    def propose(self, mc):
        return self.queue.pop(0) if self.queue else self.space.random(self.rng, mc)

    def observe(self, record):
        pass


class FixtureMonitor:
    def __init__(self, *args):
        self.calls = 0

    def update(self, record):
        self.calls += 1
        return self.calls >= 1


def embed(texts):
    mapping = {'similar': [1., 0.], 'other': [0., 1.], 'opposite': [-1., 0.]}
    return np.asarray([mapping.get(text, [1., .2]) for text in texts])


class FormulaTests(unittest.TestCase):
    def test_frame_average_not_flatten(self):
        features = np.tile([1., 2., 3.], (16, 1))
        self.assertEqual(video_feature(features).shape, (3,))
        np.testing.assert_allclose(video_feature(features), np.asarray([1., 2., 3.]) / math.sqrt(14))
        with self.assertRaises(ValueError):
            video_feature(features[:15])

    def test_mc_formula_max_per_video_then_divide_video_count(self):
        kb = VideoKnowledgeBase({'seed': 42}, {'a': [1., 0.], 'b': [.8, .6], 'target': [1., 0.]})
        kb.video_ids = ['a', 'b']
        c = cfg()
        kb.add_result('a', TuningResult(['q'], [trial(c, {'q': 1}), trial(c, {'q': 4}),
                                               trial(c, {'different': 5}, stage='query')], c))
        kb.add_result('b', TuningResult(['q'], [trial(c, {'q': 2})], c))
        expected = (similarity_weight(1.) * 4 + similarity_weight(.8) * 2) / 2
        self.assertAlmostEqual(kb.mc_value_ranking('target')[c.mc], expected)

    def test_cluster_quotas_and_determinism(self):
        features = {str(i): np.asarray([float(i // 10), i / 100.]) for i in range(40)}
        first, second = VideoKnowledgeBase({}, features), VideoKnowledgeBase({}, features)
        self.assertEqual(first.build(.2, 42), second.build(.2, 42))
        self.assertEqual(len(first.video_ids), 8)
        self.assertGreaterEqual(first.n_clusters, 2)

    def test_kb_context_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = VideoKnowledgeBase({'model': 'A', 'slo': .5, 'seed': 42}, {'v': [1., 0.]})
            kb.build()
            path = Path(folder) / 'kb.json'
            kb.save(path)
            self.assertEqual(VideoKnowledgeBase.load(path, kb.context).video_ids, kb.video_ids)
            with self.assertRaises(ValueError):
                VideoKnowledgeBase.load(path, {**kb.context, 'seed': 43})

    def test_adapter_formula_threshold_and_ttft_tie(self):
        slow, fast = cfg(4), cfg(8)
        result = TuningResult(['q1', 'q2'], [trial(slow, {'q1': 4, 'q2': 0}, {'q1': .4, 'q2': .4}),
                                           trial(fast, {'q1': 4}, {'q1': .1}, stage='query')], slow)
        adapter = OnlineSemanticAdapter({'q1': 'similar', 'q2': 'other'}, result, embed, .5)
        self.assertEqual(adapter.adapt('similar').key, fast.key)
        self.assertEqual(adapter.adapt('opposite').key, slow.key)
        # Strictly greater than threshold: similarity == 1 is rejected at threshold 1.
        adapter.threshold = 1.
        self.assertEqual(adapter.adapt('similar').key, slow.key)

    def test_duplicate_question_text_keeps_distinct_ids(self):
        a, b = cfg(4), cfg(8)
        result = TuningResult(['q1', 'q2'], [trial(a, {'q1': 5, 'q2': 1}), trial(b, {'q1': 3, 'q2': 4})], a)
        adapter = OnlineSemanticAdapter({'q1': 'similar', 'q2': 'similar'}, result, embed, .1)
        self.assertEqual(adapter.adapt('similar').key, b.key)

    def test_equation_four_divides_by_query_count_not_weight_sum(self):
        a, b = cfg(4), cfg(8)
        vectors = {'target': [1., 0.], 'close': [1., 0.], 'less_close': [.6, .8]}
        embedding = lambda texts: np.asarray([vectors[t] for t in texts])
        result = TuningResult(['q1', 'q2'], [trial(a, {'q1': 3., 'q2': 3.}),
                                           trial(b, {'q1': 2.5}, stage='query')], a)
        adapter = OnlineSemanticAdapter({'q1': 'close', 'q2': 'less_close'}, result, embedding, .1)
        self.assertEqual(adapter.adapt('target').key, b.key)

    def test_threshold_uses_kb_tuning_only(self):
        c = cfg()
        result = TuningResult(['q1', 'q2'], [trial(c, {'q1': 3, 'q2': 4})], c)
        threshold = select_similarity_threshold({'v': result}, {'v': {'q1': 'similar', 'q2': 'other'}}, embed, [0., .5, 1.])
        self.assertEqual(threshold, 0.)


class PruningTests(unittest.TestCase):
    def test_pre_pruning_and_boundaries(self):
        pruner = LatencyAwareSpacePruner(.5, profile())
        self.assertFalse(pruner.should_prune(cfg(4), ['q']))
        self.assertTrue(pruner.should_prune(cfg(128), ['q']))

    def test_dominance_holds_nonmonotone_parameters_fixed(self):
        pruner = LatencyAwareSpacePruner(1., profile())
        pruner.observe(trial(cfg(4), {'q': 5}, {'q': 1.1}))
        self.assertTrue(pruner.should_prune(cfg(8), ['q']))
        self.assertFalse(pruner.should_prune(cfg(8, 'keyframe'), ['q']))
        self.assertFalse(pruner.should_prune(cfg(8), ['unseen']))
        self.assertFalse(pruner.should_prune(cfg(8), ['q', 'unseen']))

    def test_decreasing_knob_and_retrieval_scope(self):
        p = profile()
        p.overheads.update({'object|None': .01, 'object|BERT': .02})
        pruner = LatencyAwareSpacePruner(1., p, [KnobSpec('object', 'confidence_threshold', 'decreasing')])
        lower = Configuration({'object': {'confidence_threshold': .5, 'model': 'APELD'}}, 'None', .3)
        heavier = Configuration({'object': {'confidence_threshold': .3, 'model': 'APELD'}}, 'None', .3)
        self.assertTrue(pruner.dominates(lower, heavier))
        heavier.retrieval_model = 'BERT'
        self.assertFalse(pruner.dominates(lower, heavier))

    def test_all_sweep_values_required(self):
        def measure(c):
            return {4: 10, 8: 5, 16: 20}[c.modalities['visiontoken']['num_frame']]
        self.assertEqual(list(profile_monotonic_knobs(measure, cfg(), {('visiontoken', 'num_frame'): [4, 8, 16]})), [])

    def test_partial_token_history_never_transfers_across_queries(self):
        pruner = LatencyAwareSpacePruner(.5, profile())
        record = TrialRecord(cfg(), [Observation('q', 'x', 4., .1, 900)], 'query', .1, .5)
        pruner.observe(record)
        self.assertTrue(pruner.should_prune(cfg(8), ['q']))
        self.assertFalse(pruner.should_prune(cfg(8), ['different']))

    def test_unmeasured_lengths_not_automatically_banned(self):
        pruner = LatencyAwareSpacePruner(10., profile())
        self.assertFalse(pruner.should_prune(cfg(128), ['q']))

    def test_real_measurement_profile_builder_and_scope_validation(self):
        rows = []
        for frames, ttft in [(4, .1), (8, .15), (16, .2)]:
            rows.append({'model': 'fixture-model', 'serving_context': {'batch_size': 1},
                         'config': asdict(cfg(frames)), 'video_id': 'v', 'question_id': 'q',
                         'token_count': frames * 10, 'modality_tokens': {'visiontoken': frames * 10},
                         'ttft': ttft + .01, 'timing': {'llm_ttft': ttft}})
        measured = build_profile(rows, 'fixture-model', {'batch_size': 1}, tiny_space())
        self.assertEqual(measured.vt_tokens_per_frame, 10.)
        self.assertEqual(measured.monotone_knobs, [KnobSpec('visiontoken', 'num_frame', 'increasing')])
        with self.assertRaises(ValueError):
            build_profile(rows, 'other-model', {'batch_size': 1}, tiny_space())
        rows[1]['token_count'] = 20
        measured = build_profile(rows, 'fixture-model', {'batch_size': 1}, tiny_space())
        self.assertEqual(measured.monotone_knobs, [])


class TuningTests(unittest.TestCase):
    def test_optuna_public_api_restricted_studies_and_pruning_feedback(self):
        studies = []
        class PublicTrial:
            def __init__(self, params):
                self.params = params
            def suggest_categorical(self, key, choices):
                value = self.params.get(key, choices[0])
                if value not in choices:
                    raise ValueError('Enqueued value outside the declared distribution')
                return value
        class PublicStudy:
            def __init__(self):
                self.queue, self.feedback = [], []
            def enqueue_trial(self, params):
                self.queue.append(params)
            def ask(self):
                return PublicTrial(self.queue.pop(0) if self.queue else {})
            def tell(self, *args, **kwargs):
                self.feedback.append((args, kwargs))
        def create_study(**kwargs):
            study = PublicStudy()
            studies.append(study)
            return study
        optuna = SimpleNamespace(create_study=create_study, samplers=SimpleNamespace(TPESampler=lambda **kw: kw),
                                 trial=SimpleNamespace(TrialState=SimpleNamespace(PRUNED='PRUNED')))
        space = SearchSpace({'visiontoken': {'num_frame': [4, 8], 'sampling_method': ['uniform']},
                             'caption': {'seconds_per_caption': [1], 'frames_per_caption': [1], 'model': ['llava']}}, ['None'], [0.])
        with patch.dict('sys.modules', {'optuna': optuna}):
            search = TPESearch(space, 42)
            search.enqueue(cfg(8), ('visiontoken',))
            first = search.propose(('visiontoken',))
            self.assertEqual(first.mc, ('visiontoken',))
            search.observe(trial(first, {'q': 4.}))
            second = search.propose(('caption',))
            self.assertEqual(second.mc, ('caption',))
            search.observe(None)
        self.assertEqual(len(studies), 2)
        self.assertEqual(studies[1].feedback[0][1]['state'], 'PRUNED')

    def test_missing_optuna_is_an_explicit_error(self):
        with patch.dict('sys.modules', {'optuna': None}):
            with self.assertRaisesRegex(RuntimeError, 'Optuna'):
                TPESearch(tiny_space(), 42)

    def test_emmr_short_budget_window_and_errors_are_not_suppressed(self):
        capture = {}
        class Trial:
            def suggest_float(self, *args):
                return 0.
        class Study:
            def enqueue_trial(self, values):
                capture['values'] = values
            def ask(self):
                return Trial()
            def tell(self, trial, quality):
                capture['quality'] = quality
        def median(evaluator, **kwargs):
            capture['median'] = kwargs
            return object()
        def terminator(**kwargs):
            capture['terminator'] = kwargs
            return SimpleNamespace(should_terminate=lambda study: True)
        modules = {'optuna': SimpleNamespace(create_study=lambda **kwargs: Study()),
                   'optuna.terminator': SimpleNamespace(EMMREvaluator=lambda **kwargs: object(),
                                                        MedianErrorEvaluator=median, Terminator=terminator)}
        with patch.dict('sys.modules', modules):
            monitor = EMMRMonitor(tiny_space(), 42)
            self.assertTrue(monitor.update(trial(cfg(), {'q': 4.})))
            self.assertEqual(capture['median'], {'warm_up_trials': 3, 'n_initial_trials': 5})
            self.assertEqual(capture['quality'], 4.)
            monitor.terminator.should_terminate = lambda study: (_ for _ in ()).throw(RuntimeError('EMMR failed'))
            with self.assertRaisesRegex(RuntimeError, 'EMMR failed'):
                monitor.update(trial(cfg(), {'q': 4.}))

    def test_per_request_penalty_even_when_mean_latency_fits(self):
        record = trial(cfg(), {'a': 5, 'b': 5}, {'a': .1, 'b': 1.1})
        self.assertLess(record.ttft, 1.)
        self.assertEqual(record.quality, 2.5)

    def test_subset_only_and_missing_not_zero(self):
        backend = FixtureBackend()
        objective = Objective(backend, ['v_0', 'v_1'], 1.)
        record = objective.evaluate_config(cfg(), ['v_1'], 'query')
        self.assertEqual(backend.calls, [['v_1']])
        self.assertEqual(set(record.scores), {'v_1'})
        self.assertEqual(record.budget_cost, .5)
        with self.assertRaises(ValueError):
            objective.evaluate_config(cfg(), ['held_out'])

    def test_bad_backend_cannot_return_full_query_set(self):
        backend = FixtureBackend()
        backend.evaluate = lambda c, ids: [Observation('unrequested', 'x', 4., .1)]
        with self.assertRaises(ValueError):
            Objective(backend, ['v_0', 'v_1'], 1.).evaluate_config(cfg(), ['v_0'])

    def test_fractional_budget_stage_transition_and_shared_pruner(self):
        space = tiny_space()
        backend = FixtureBackend()
        pruner = LatencyAwareSpacePruner(1., profile())
        objective = Objective(backend, ['v_0', 'v_1'], 1., pruner)
        tpe = FixtureTPE(space, 42)
        tpe.queue = [cfg(4)]
        tuner = TwostageTuner(space, 42, tpe=tpe, monitor=FixtureMonitor())
        result = tuner.tune(objective, 3)
        self.assertEqual(backend.calls[0], ['v_0', 'v_1'])
        self.assertTrue(all(call == ['v_1'] for call in backend.calls[1:]))
        self.assertLessEqual(result.budget_used, 3)
        self.assertIn('emmr', result.switch_reason)
        self.assertEqual(len(pruner.token_observations), sum(len(c) for c in backend.calls))

    def test_empty_poor_subset_stops_without_extra_budget(self):
        space = tiny_space()
        objective = Objective(FixtureBackend(), ['v_0', 'v_1'], 1.)
        tpe = FixtureTPE(space, 42)
        tpe.queue = [cfg(8)]
        result = TwostageTuner(space, tpe=tpe, monitor=FixtureMonitor()).tune(objective, 30)
        self.assertEqual(result.budget_used, 1.)
        self.assertEqual(result.switch_reason, 'emmr:no_poor_queries')

    def test_budget_allocation_conserves_total(self):
        for values in ({('a',): 5., ('b',): 2.}, {('a',): 0., ('b',): 0.}):
            self.assertEqual(sum(allocate_budget(30, values).values()), 30)

    def test_baselines_have_real_distinct_feedback_paths(self):
        space = tiny_space()
        histories = [trial(cfg(4 + i % 2 * 4), {'v_0': 5, 'v_1': 2 + i % 3},
                           {'v_0': .1, 'v_1': 1.5 if i % 2 else .2}) for i in range(8)]
        candidate = SCOOTSearch(space, np.random.default_rng(42)).propose(histories, ['v_0', 'v_1'])
        self.assertIn(candidate.modalities['visiontoken']['num_frame'], [4, 8, 16])
        op = OPPerTuneSearch(space, np.random.default_rng(42))
        first = op.propose([], ['v_0'])
        op.observe(trial(first, {'v_0': 5}))
        second = op.propose([], ['v_0'])
        op.observe(trial(second, {'v_0': 1}))
        self.assertIsNone(op.pending)
        self.assertTrue(any(s['center'] != .5 for s in op.numeric.values()))
        for method in ('VT', 'Random', 'SCOOT', 'LLMConf', 'OPPerTune'):
            objective = Objective(FixtureBackend(), ['v_0', 'v_1'], 1.)
            result = tune_baseline(method, objective, space, 8, 42)
            self.assertLessEqual(result.budget_used, 8)
            self.assertTrue(all(set(t.scores) == {'v_0', 'v_1'} for t in result.history))

    def test_split_is_paired_deterministic_and_disjoint(self):
        video = Video('v', Path('unused'), [{'question_id': f'v_{i}'} for i in range(20)])
        a, b = split_queries(video, .4, 42)
        self.assertEqual((a, b), split_queries(video, .4, 42))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a), 8)
        self.assertNotEqual(a, split_queries(video, .4, 43)[0])

    def test_history_round_trip_and_no_pickle_guessing(self):
        result = TuningResult(['q'], [trial(cfg(), {'q': 4})], cfg())
        self.assertEqual(TuningResult.from_dict(json.loads(json.dumps(result.to_dict()))).to_dict(), result.to_dict())
        with self.assertRaises(ValueError):
            TuningResult.from_dict({'trail_history': []})


class ExportTests(unittest.TestCase):
    def make_run(self, folder, seeds=(42, 43), completed=(42, 43)):
        settings = {'models': ['M'], 'datasets': {'D': {}}, 'slos': [1.], 'methods': ['Prysma'], 'seeds': list(seeds)}
        atomic_json(Path(folder) / 'manifest.json', {'settings': settings, 'fingerprint': 'fixture',
                    'datasets': {'D': {'videos': {'v': ['q0', 'q1']}, 'qa_digest': 'qa'}}})
        for seed in completed:
            raw = 2. if seed == 42 else 4.
            context = {'model': 'M', 'dataset': 'D', 'slo': 1., 'seed': seed, 'fingerprint': 'fixture', 'qa_digest': 'qa'}
            path = context_path(folder, 'M', 'D', 1., seed) / 'Prysma/videos/v.json'
            atomic_json(path, {'complete': True, 'context': context, 'method': 'Prysma', 'predictions': [
                {'question_id': 'q0', 'raw_score': raw, 'ttft': .1, 'quality': raw},
                {'question_id': 'q1', 'raw_score': 5., 'ttft': 1.1, 'quality': 0.}]})

    def test_question_weighting_and_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_run(folder)
            seeds, summary = export_results(folder)
            self.assertEqual([r['quality'] for r in seeds], [1., 2.])
            self.assertEqual(summary[0]['mean'], 1.5)
            self.assertAlmostEqual(summary[0]['std'], math.sqrt(.5))

    def test_missing_seed_is_incomplete_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_run(folder, completed=(42,))
            seeds, summary = export_results(folder)
            self.assertIsNone(seeds[1]['quality'])
            self.assertIsNone(summary[0]['mean'])
            self.assertFalse(summary[0]['complete'])

    def test_full_pipeline_with_independent_fixture_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = load_settings(ROOT / 'configs/main_result.json')
            settings.update(models=['LLaVA-Video-7B'], datasets={'MSVD-QA': {'video_limit': 2, 'train_ratio': .5}},
                            slos=[1.], seeds=[42, 43], budget=3)
            settings['runtime']['output_root'] = str(Path(folder) / 'results')
            videos = [Video(v, Path('unused'), [{'video_id': v, 'question_id': f'{v}_{i}', 'question': 'similar', 'answer': 'reference'}
                                              for i in range(4)]) for v in ('a', 'b')]
            p = profile(default_space('MSVD-QA'))
            p.model = 'LLaVA-Video-7B'
            p.serving_context = settings['runtime']['serving_context']
            p.token_counts, p.ttfts, p.vt_tokens_per_frame = [0, 10000], [0., .5], 1.
            with patch('experiments.run_main_result.load_videos', return_value=videos), \
                 patch('experiments.run_main_result.VideoBackend', FixtureBackend), \
                 patch('experiments.run_main_result.SentenceEmbedder', return_value=embed), \
                 patch('experiments.run_main_result.TokenLatencyModel.load', return_value=p), \
                 patch('config_tuner.config_tuner.TPESearch', FixtureTPE), \
                 patch('config_tuner.config_tuner.EMMRMonitor', FixtureMonitor):
                run(settings, ROOT / 'configs/main_result.json')
                # A second invocation must resume without regenerating completed results.
                run(settings, ROOT / 'configs/main_result.json')
            seeds, summary = export_results(settings['runtime']['output_root'])
            self.assertEqual(len(seeds), 12)
            self.assertEqual(len(summary), 6)
            self.assertTrue(all(r['complete'] for r in summary))


class StaticChecks(unittest.TestCase):
    def test_all_python_files_parse(self):
        files = list(ROOT.rglob('*.py'))
        self.assertGreater(len(files), 40)
        for file in files:
            ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))

    def test_core_modules_import_without_gpu_packages(self):
        for module in ('config_tuner.records', 'config_tuner.space', 'config_tuner.knowledgebase',
                       'config_tuner.latency_pruner', 'config_tuner.adapter', 'config_tuner.objective',
                       'config_tuner.config_tuner', 'config_tuner.baselines', 'experiments.run_main_result',
                       'experiments.export_main_result', 'experiments.profile_latency'):
            self.assertIsNotNone(importlib.import_module(module))

    def test_default_figure_matrix_size(self):
        s = load_settings(ROOT / 'configs/main_result.json')
        self.assertEqual(len(s['models']) * len(s['datasets']) * len(s['slos']) * len(s['methods']), 108)
        self.assertEqual(len(s['seeds']), 5)

    def test_backend_bridge_passes_exact_subset_and_measured_timing(self):
        settings = load_settings(ROOT / 'configs/main_result.json')
        video = Video('v', Path('unused'), [{'question_id': 'v_0', 'question': 'similar', 'answer': 'ref'},
                                          {'question_id': 'v_1', 'question': 'other', 'answer': 'ref'}])
        backend = VideoBackend(settings['runtime'], 'LLaVA-Video-7B', 'MSVD-QA', video, 'fixture')
        capture = {}
        def infer(**kwargs):
            capture.update(kwargs)
            return [{'question_id': 'v_1', 'pred': 'fixture answer', 'ttft': .2, 'token_count': 100,
                     'keyword_extraction_time': .05, 'object_retrieve_time': .03}]
        legacy = SimpleNamespace(inference_parameter=SimpleNamespace(), run_inference_for_one_video=infer)
        c = Configuration({'object': {'fps': 1, 'confidence_threshold': .5, 'model': 'RTDETR'}}, 'BERT', .3)
        with patch.object(backend, 'prepare', return_value=c.modalities), patch.object(backend, '_inference', return_value=legacy):
            rows = backend.measure(c, ['v_1'])
        self.assertEqual(capture['question_ids'], ['v_1'])
        self.assertEqual(capture['keyword_extractor_name'], 'KeyBERT')
        self.assertAlmostEqual(rows[0]['ttft'], .28)
        self.assertTrue(legacy.inference_parameter.force_inference)
        self.assertFalse(legacy.inference_parameter.use_existing_retrieve_result)


if __name__ == '__main__':
    unittest.main()
