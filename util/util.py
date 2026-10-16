"""Shared runtime paths, modality caches and retrieval helpers."""
import json
import os
from pathlib import Path
import pickle
import random
import socket
import time

import numpy as np
import torch
from decord import VideoReader, cpu
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from util.preprocess_constant import Modality, LLMName, KeyWordExtractorNames, modality2priority
from util.runtime_config import configuration, dataset_path

LLM_INF = 100000


id2category = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


def sample_video_frames(video_path, num_frame):
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    duration = total_frame_num / vr.get_avg_fps()
    interval = round(total_frame_num / num_frame)
    frame_idxs = [i for i in range(0, len(vr), interval)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idxs]
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idxs).asnumpy()

    return spare_frames[0:num_frame], frame_idxs[0:num_frame]


def config2path_load(config):  # config转化为路径字符串
    escape_keys = ['modality', 'frame_idx', 'text_prompt']
    return config2path(config, escape_keys)


def config2path(config, escape_keys):
    import hashlib
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    value = {k: v for k, v in value.items() if k not in escape_keys}
    return 'cfg_' + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def config2path_preprocess(config):  # config转化为路径字符串
    escape_keys = ['modality', 'retrieve_threshold', 'retrieval_model']
    return config2path(config, escape_keys)


def preprocess_store_path(modality, config, video_name, dataset_name, llm_name=None):
    home_dir = configuration()['preprocess_root']
    if config is not None:  # modality
        store_dir = os.path.join(home_dir, dataset_name, video_name, modality)
        if modality == Modality.VisionToken:
            assert llm_name is not None
            store_dir = os.path.join(home_dir, dataset_name, video_name, f'{modality}_{llm_name}')
        os.makedirs(store_dir, exist_ok=True)
        store_path = os.path.join(store_dir, config2path_preprocess(config))
    else:  # video meta信息
        store_dir = os.path.join(home_dir, dataset_name, video_name)
        os.makedirs(store_dir, exist_ok=True)
        store_path = store_dir
    return store_path


def extract_frame_config(config):
    return OmegaConf.create({
        'num_frame': config['num_frame'],
        'sampling_method': config['sampling_method']
    })


def get_inference_result_file_name(modality_config: dict):
    modalities = list(modality_config.keys())
    modalities = sorted(modalities, key=lambda x: modality2priority[x])
    inference_result_file_name = ''
    for modality in modalities:
        config_path = config2path_load(modality_config[modality])
        if inference_result_file_name == '':
            inference_result_file_name = '_'.join([modality, config_path])
        else:
            inference_result_file_name = '_'.join([inference_result_file_name, modality, config_path])
    return inference_result_file_name


def get_inference_result_dir(llm_name, dataset_name, video_id, modality_config: dict, keyword_extractor_name):
    # modalities = list(modality_config.keys())
    # modalities = sorted(modalities, key=lambda x: modality2priority[x])
    # inference_result_file_name = ''
    # for modality in modalities:
    #     config_path = config2path_load(modality_config[modality])
    #     if inference_result_file_name == '':
    #         inference_result_file_name = '_'.join([modality, config_path])
    #     else:
    #         inference_result_file_name = '_'.join([inference_result_file_name, modality, config_path])

    # if not has_retrieved_modality(modality_config):  # todo 这里默认用LLM extractor作为不需要retrieve的modality
    #     keyword_extractor_name = KeyWordExtractorNames.LLMExtractor
    inference_result_file_name = get_inference_result_file_name(modality_config)
    inference_result_dir = os.path.join(configuration()['cache_root'], 'legacy', llm_name, dataset_name, 'inference_result',
                                        video_id,
                                        keyword_extractor_name)
    os.makedirs(inference_result_dir, exist_ok=True)
    inference_result_dir = os.path.join(inference_result_dir, inference_result_file_name)
    return inference_result_dir


def get_video_path(dataset_name, video_name):
    return os.path.join(dataset_path(dataset_name), 'videos', video_name)


def caption2prompt(caption: dict):  # 按照LLoVi的方法把caption转化成输入的prompt
    caption_prompt = []
    for i, cap in enumerate(caption.values()):
        caption_prompt.append(f'{i}: {cap}\n')
    return caption_prompt


def set_random_seed(seed=42):
    global np_state
    global rd_state
    random.seed(seed)
    np.random.seed(seed)
    np_state = np.random.get_state()
    rd_state = random.getstate()
    torch.manual_seed(seed)
    print(f'set random seed {seed}')


def get_video_meta_path(video_name, dataset_name):
    return os.path.join(preprocess_store_path(None, None, video_name, dataset_name), 'video_meta')


def get_retrieve_file_name(modality, config=None):
    if config is not None:
        file_name = f'{modality}_retrieve_result_{config2path_load(config)}.json'
    else:
        file_name = f'{modality}_retrieve_result.json'
    return file_name


def load_retrieve_result(llm_name, dataset_name, video_id, modality, keyword_extractor, config=None):
    retrieve_result_dir = os.path.join(configuration()['cache_root'], 'legacy', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_retrieve_file_name(modality, config)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    if os.path.exists(retrieve_result_file):
        return json.load(open(retrieve_result_file, 'r'))
    else:
        return {}


def get_keyword_file_name(keyword_extractor):
    file_name = 'keyword_result.json'
    return file_name


def save_keyword_result(llm_name, keyword_result, dataset_name, video_id, keyword_extractor):
    retrieve_result_dir = os.path.join(configuration()['cache_root'], 'legacy', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_keyword_file_name( keyword_extractor.name)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    json.dump(keyword_result, open(retrieve_result_file, 'w'))


def load_keyword_result(llm_name, dataset_name, video_id, keyword_extractor):
    retrieve_result_dir = os.path.join(configuration()['cache_root'], 'legacy', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_keyword_file_name(keyword_extractor.name)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    if os.path.exists(retrieve_result_file):
        return json.load(open(retrieve_result_file, 'r'))
    else:
        return {}


def encode_sentences(sentence_list, model_name, sentence_model, tokenize_func):
    '''given a list of sentences, return the embeddings for them using the sentence encoder model'''
    assert model_name == 'clip'
    emb_list = []
    device = configuration()['device']
    num_iter = 1
    with torch.no_grad():
        for sentence in sentence_list:
            # emb_list.append(sentence_model.encode_text(clip.tokenize([sentence]).to(device)).cpu().numpy())
            emb_list.append(sentence_model.encode_text(tokenize_func([sentence]).to(device)).cpu().numpy())
    emb_list = np.concatenate(emb_list, axis=0)
    return emb_list


def retrieve_candidate_objects(description, uid2emb, uid2category, sentence_model, tokenize_func,
                               retrieve_threshold=0.26):
    def select_elements_with_repetition(lst, count):
        if not lst:  # 如果列表为空，返回空列表
            return []

        result = []
        for i in range(count):
            # 使用模运算来循环获取元素
            result.append(lst[i % len(lst)])
        return result

    def compute_cosine_similarity(target_embedding, embedding_list):
        embedding_list_np = [embedding.numpy() for embedding in embedding_list]
        if len(embedding_list_np) == 0:
            return np.array([])
        target_embedding_tensor = target_embedding.reshape(1, -1)
        # Compute cosine similarity
        start = time.monotonic()
        similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list_np)
        end = time.monotonic()
        return similarity_scores.reshape(-1)

    des_emb = encode_sentences([f"a photo of a {description}."], model_name='clip', sentence_model=sentence_model,
                               tokenize_func=tokenize_func)
    scores = compute_cosine_similarity(des_emb, list(uid2emb.values()))
    indices = np.where(scores >= retrieve_threshold)[0]
    candidate_uids = []
    for i in indices:
        candidate_uids.append(list(uid2emb)[i])
    return candidate_uids, scores


def get_retrieved_object_result(object_result, keywords, uid2clip, uid2category, config, tokenize_func, sentence_model,
                                use_org_label=False):
    # retrieve_threshold = 0.26
    retrieve_threshold = 0.26
    if 'retrieve_threshold' in config:
        retrieve_threshold = config['retrieve_threshold']
    start_time = time.monotonic()
    candidate_uids = set()
    uid2_keyword = {}
    if keywords is None:  # 返回所有的object 通过设置retrieve_threshold = 0 实现
        uid2_keyword = uid2category
        uids, retrieve_scores = retrieve_candidate_objects('object', uid2clip, uid2category, sentence_model,
                                                           tokenize_func, retrieve_threshold=0.0)
        candidate_uids.update(uids)
    else:
        for keyword in keywords:
            uids, retrieve_scores = retrieve_candidate_objects(keyword, uid2clip, uid2category, sentence_model,
                                                               tokenize_func, retrieve_threshold=retrieve_threshold)
            candidate_uids.update(uids)
            for uid in uids:
                uid2_keyword[uid] = keyword
    end_time = time.monotonic()
    retrieved_object_result = []
    for frame_result_str in object_result:
        frame_result = frame_result_str.strip().split(';')
        retrieved_frame_result = ''
        for object_str in frame_result:
            if ':' not in object_str:
                continue
            org_category_name, bbox = object_str.split(':', 1)
            org_category_name = org_category_name.strip()
            uid = int(org_category_name.split('_')[-1])
            if uid in candidate_uids:
                category_name = org_category_name.split('_')[0] if use_org_label else uid2_keyword[uid]
                retrieved_frame_result += f'{category_name} / {org_category_name}: {bbox}; '
        if len(retrieved_frame_result) > 0:
            retrieved_frame_result = retrieved_frame_result[:-2]
        if len(retrieved_frame_result) > 0:
            retrieved_object_result.append(retrieved_frame_result)
    retrieve_time = end_time - start_time
    return retrieved_object_result, retrieve_time


class InferenceParameter:
    def __init__(self, llm_name=LLMName.LlavaVideoQwen7b):
        self.use_existing_retrieve_result = False
        self.use_existing_keyword_result = False
        self.do_inference = True
        self.need_save_inference_result = True
        self.need_save_retrieve_result = True
        self.need_save_keyword_result = True
        self.force_inference = True
        self.llm_name = llm_name

np_state = None
rd_state = None
set_random_seed(42)
server_config = OmegaConf.create({m: {'port': p} for m, p in configuration()['server_ports'].items()})
warm_up_video_path = get_video_path('sample_video', '2440175990.mp4')
token_margin = 100
