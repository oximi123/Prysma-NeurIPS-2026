"""Real extraction/inference/judge bridge. Importing this module loads no models."""
from __future__ import annotations
from dataclasses import asdict
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import time
from urllib.request import Request, urlopen
import numpy as np

from config_tuner.knowledgebase import video_feature
from config_tuner.records import Observation, canonical

MODEL_ALIASES = {'LLaVA-Video-7B': 'LlavaVideoQwen7b', 'VideoLLaMA2-7B': 'VideoLlamaMistral7b'}


class SentenceEmbedder:
    def __init__(self, model_name, device):
        self.model_name, self.device = model_name, device
        self.model, self.cache = None, {}

    def __call__(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
            vectors = self.model.encode(missing, normalize_embeddings=True, convert_to_numpy=True)
            self.cache.update(zip(missing, vectors))
        return np.asarray([self.cache[t] for t in texts])


class QualityJudge:
    def __init__(self, settings):
        self.settings = settings

    def score(self, question, reference, answer):
        key = os.environ.get(self.settings['api_key_env'])
        if not key:
            raise RuntimeError(f"Set {self.settings['api_key_env']} for the configured judge")
        messages = [
            {'role': 'system', 'content': 'Evaluate the correctness of a video question-answer prediction. '
             'Compare meaning with the reference; accept synonyms and paraphrases. Return only a JSON '
             'object with a numeric score in [0,5], where 5 is fully correct, and pred as yes or no.'},
            {'role': 'user', 'content': f'Question: {question}\nReference answer: {reference}\nPredicted answer: {answer}'},
        ]
        body = json.dumps({'model': self.settings['model'], 'messages': messages, 'temperature': 0}).encode()
        request = Request(self.settings['api_base'].rstrip('/') + '/chat/completions', body,
                          {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        for attempt in range(self.settings.get('max_attempts', 3)):
            try:
                with urlopen(request, timeout=self.settings.get('timeout_seconds', 120)) as response:
                    text = json.load(response)['choices'][0]['message']['content'].strip()
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    result = ast.literal_eval(text)
                score = float(result['score'])
                if not np.isfinite(score) or not 0 <= score <= 5:
                    raise ValueError('Judge score outside [0,5]')
                return score
            except Exception as exc:
                if attempt + 1 == self.settings.get('max_attempts', 3):
                    raise RuntimeError('Judge request failed; no score was imputed') from exc
                time.sleep(min(2 ** attempt, 4))


class VideoBackend:
    def __init__(self, runtime, model, dataset, video, run_fingerprint):
        self.runtime, self.model, self.dataset, self.video = runtime, model, dataset, video
        self.alias = MODEL_ALIASES[model]
        self.questions = {q['question_id']: q for q in video.questions}
        self.judge = QualityJudge(runtime['judge'])
        self.cache = Path(runtime['cache_root']) / run_fingerprint / model / dataset / video.video_id
        self.legacy = None
        self.meta = None

    def _metadata(self):
        if self.meta is None:
            from decord import VideoReader, cpu
            vr = VideoReader(str(self.video.path), ctx=cpu(), num_threads=1)
            self.meta = {'duration': len(vr) / vr.get_avg_fps(), 'fps': vr.get_avg_fps(), 'total_frames': len(vr)}
            from util.util import get_video_meta_path
            path = Path(get_video_meta_path(self.video.path.name, self.dataset))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as stream:
                pickle.dump(self.meta, stream)
        return self.meta

    def _request(self, modality, config):
        from util.util import preprocess_store_path
        cfg = {**config, 'modality': modality}
        path = Path(preprocess_store_path(modality, cfg, self.video.path.name, self.dataset,
                                         self.alias if modality == 'visiontoken' else None))
        if path.is_file():
            return path
        payload = (self.dataset, self.video.path.name, cfg)
        if modality == 'visiontoken':
            payload += (self.alias,)
        elif modality == 'object':
            payload += (None,)
        with socket.create_connection((self.runtime['server_host'], self.runtime['server_ports'][modality]),
                                      timeout=self.runtime.get('socket_timeout_seconds', 3600)) as client:
            client.sendall(pickle.dumps(payload))
            response = bytearray()
            while True:
                part = client.recv(65536)
                if not part:
                    break
                response.extend(part)
        returned = Path(pickle.loads(response))
        if returned.resolve() != path.resolve() or not returned.is_file():
            raise RuntimeError(f'{modality} extraction server returned an unexpected/missing cache path')
        return path

    def feature(self):
        self._metadata()
        cfg = {'num_frame': 16, 'sampling_method': 'uniform'}
        self._request('frame', cfg)
        path = self._request('clip', cfg)
        with path.open('rb') as stream:
            features = pickle.load(stream)['clip']
        if hasattr(features, 'detach'):
            features = features.detach().float().cpu().numpy()
        return video_feature(features)

    def prepare(self, config):
        meta = self._metadata()
        modalities = copy.deepcopy(config.modalities)
        for modality, cfg in modalities.items():
            if modality == 'object':
                cfg['num_frame'] = max(1, int(meta['duration'] * min(cfg['fps'], meta['fps'])))
                cfg['sampling_method'] = 'uniform'
            if modality in ('visiontoken', 'ocr'):
                self._request('frame', {'num_frame': cfg['num_frame'], 'sampling_method': cfg['sampling_method']})
            self._request(modality, cfg)
            if modality in ('object', 'ocr', 'asr'):
                cfg['retrieve_threshold'] = config.retrieval_threshold
        return modalities

    def _inference(self):
        if self.legacy is None:
            import script_run_inference as legacy
            self.legacy = legacy
        if self.legacy.model is None or self.legacy.inference_parameter.llm_name != self.alias:
            self.legacy.model = None
            self.legacy.inference_parameter.llm_name = self.alias
            self.legacy.load_llm()
        return self.legacy

    def measure(self, config, question_ids):
        """Measure real inference without requesting quality judgments (calibration)."""
        modalities = self.prepare(config)
        legacy = self._inference()
        # The scored observation cache is the authoritative cache. Do not pick up
        # historical positional/whole-video inference results in a new experiment.
        legacy.inference_parameter.force_inference = True
        legacy.inference_parameter.need_save_inference_result = False
        legacy.inference_parameter.use_existing_retrieve_result = False
        legacy.inference_parameter.use_existing_keyword_result = False
        legacy.inference_parameter.need_save_retrieve_result = False
        legacy.inference_parameter.need_save_keyword_result = False
        extractor_name = {'BERT': 'KeyBERT', 'LLM': self.alias, 'None': 'Bare'}[config.retrieval_model]
        results = legacy.run_inference_for_one_video(
            video_id2questions={self.video.video_id: self.video.questions}, dataset_name=self.dataset,
            question_type='OE', video_id2name={self.video.video_id: self.video.path.name},
            video_id=self.video.video_id, inference_parameter=legacy.inference_parameter, llm_name=self.alias,
            valid_modality2config=[modalities], keyword_extractor_name=extractor_name, question_ids=question_ids)
        measurements = []
        for result in results:
            timing = {k: float(result.get(k) or 0) for k in ('keyword_extraction_time', 'asr_retrieve_time',
                      'ocr_retrieve_time', 'object_retrieve_time', 'visiontoken_load_time', 'caption_load_time')}
            if not set(config.mc).intersection(('object', 'asr', 'ocr')):
                timing['keyword_extraction_time'] = 0.0
            timing['llm_ttft'] = float(result['ttft'])
            measurements.append({'question_id': result['question_id'], 'video_id': self.video.video_id,
                                 'answer': result['pred'], 'ttft': sum(timing.values()),
                                 'token_count': result['token_count'], 'modality_tokens': result.get('modality_tokens', {}),
                                 'timing': timing, 'config': asdict(config)})
        if len(measurements) != len(question_ids) or {m['question_id'] for m in measurements} != set(question_ids):
            raise RuntimeError('Incomplete calibration inference')
        return measurements

    def evaluate(self, config, question_ids):
        if len(set(question_ids)) != len(question_ids) or not set(question_ids).issubset(self.questions):
            raise ValueError('Unknown or duplicate evaluation query IDs')
        cached, missing = {}, []
        for q in question_ids:
            path = self.cache / config.key / (hashlib.sha256(q.encode()).hexdigest() + '.json')
            if path.is_file():
                cached[q] = Observation(**json.loads(path.read_text(encoding='utf-8')))
            else:
                missing.append(q)
        if missing:
            results = self.measure(config, missing)
            for result in results:
                q = result['question_id']
                if q not in missing:
                    raise ValueError('Inference returned an unrequested query')
                score = self.judge.score(self.questions[q]['question'], self.questions[q]['answer'], result['answer'])
                observation = Observation(q, result['answer'], score, result['ttft'], result['token_count'],
                                          result.get('modality_tokens', {}), result['timing'])
                cached[q] = observation
                path = self.cache / config.key / (hashlib.sha256(q.encode()).hexdigest() + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(asdict(observation)), encoding='utf-8')
        if set(cached) != set(question_ids):
            raise RuntimeError('Incomplete inference/judge results; no values were imputed')
        return [cached[q] for q in question_ids]
