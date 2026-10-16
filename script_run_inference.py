from util.runtime_config import configuration, model_path
from util.cached_video import install_cached_video_api, prepare_cached_inputs
import os

import torch

from util.util import LLM_INF, get_inference_result_dir, get_retrieve_file_name, get_retrieved_object_result, \
    preprocess_store_path, extract_frame_config, server_config, InferenceParameter, load_keyword_result, \
    load_retrieve_result, get_video_meta_path, save_keyword_result, token_margin
from videollama2 import DEFAULT_VIDEO_TOKEN, tokenizer_multimodal_token, KeywordsStoppingCriteria, model_init

# Device selection comes from configs/main_result.json.
from keyword_extractor.keywordextractor import BareKeywordExtractor, \
    KeywordExtractorBuilder, NonKeywordExtractor, BaseKeywordExtractor

import warnings

from util.preprocess_constant import *
import time
warnings.filterwarnings("ignore")
from vidrag_pipeline.rag_retriever_dynamic import retrieve_documents_with_dynamic
from vidrag_pipeline.scene_graph import generate_scene_graph_description

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import numpy as np
import json
import os
import ast
import socket
import pickle
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from util.prompt_factory import PromptFactory
import clip

max_new_tokens = 512
pf = PromptFactory()
sentence_model = None


def get_det_docs(port, config, question, dataset_name, video_name):
    res = []
    frame_idx = config['frame_idx']
    if len(frame_idx) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((configuration()['server_host'], port))
        data = (dataset_name, video_name, config, question)
        client_socket.send(pickle.dumps(data))
        res_list_path = pickle.loads(client_socket.recv(4096))
        res = pickle.load(open(res_list_path, 'rb'))
    return res


def det_preprocess(det_docs, location, relation, number, keyword_extractor):
    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split("/")[0].strip()
                obj_bbox = objs.split(":")[-1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)

    return scene_descriptions


device = configuration()['device']
model, tokenizer, image_processor = None, None, None
conv_template = None
max_context_length = 32768


def load_llm():
    global model, tokenizer, image_processor, conv_template, max_context_length, inference_parameter
    if inference_parameter.llm_name == LLMName.LlavaVideoQwen7b:
        overwrite_config = {}
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path(LLMName.LlavaVideoQwen7b),
            None,
            "llava_qwen",
            torch_dtype="bfloat16",
            device_map=device,
            overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
        assert isinstance(model, LlavaQwenForCausalLM)
        model.eval()
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    elif inference_parameter.llm_name == LLMName.VideoLlamaMistral7b:
        checkpoint = model_path(LLMName.VideoLlamaMistral7b)
        model, image_processor, tokenizer = model_init(checkpoint)
    else:
        raise Exception(f"No such model {inference_parameter.llm_name}")
    install_cached_video_api(model, inference_parameter.llm_name)


def tokenize_func(video_embedding, qs):
    global inference_parameter, model, tokenizer
    if inference_parameter.llm_name == LLMName.LlavaVideoQwen7b:
        if video_embedding is not None:
            question = DEFAULT_IMAGE_TOKEN + qs
        else:
            question = qs
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).to(device)
    elif inference_parameter.llm_name == LLMName.VideoLlamaMistral7b:
        if video_embedding is None:
            modal = 'text'
        else:
            modal = 'video'

        if modal == 'image':
            modal_token = DEFAULT_IMAGE_TOKEN
        elif modal == 'video':
            modal_token = DEFAULT_VIDEO_TOKEN
        elif modal == 'text':
            modal_token = ''
        else:
            raise ValueError(f"Unsupported modal: {modal}")

        if isinstance(qs, str):
            message = [{'role': 'user', 'content': modal_token + '\n' + qs}]
        elif isinstance(qs, list):
            message = copy.deepcopy(qs)
            message[0]['content'] = modal_token + '\n' + message[0]['content']
        else:
            raise ValueError(f"Unsupported type of instruct: {type(qs)}")

        if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
            system_message = [
                {'role': 'system', 'content': (
                    """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                    """\n"""
                    """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
                 }
            ]
        else:
            system_message = []

        message = system_message + message
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(
            0).long().to(device)
    else:
        raise Exception(f"No such model {inference_parameter.llm_name}")
    install_cached_video_api(model, inference_parameter.llm_name)
    return input_ids


def llava_inference_splited(qs, video_embedding=None):
    input_ids = tokenize_func(video_embedding, qs)
    assert isinstance(model, LlavaQwenForCausalLM)

    torch.cuda.synchronize(model.device)
    start = time.monotonic()
    if video_embedding is not None:
        cont = model.generate_with_video_embed(input_ids,
                                               video_embedding,
                                               modalities=["video"],
                                               do_sample=False,
                                               temperature=0,
                                               max_new_tokens=max_new_tokens,
                                               )
    else:
        cont = model.generate(input_ids,
                              do_sample=False,
                              temperature=0,
                              max_new_tokens=max_new_tokens,
                              )
    torch.cuda.synchronize(model.device)
    end = time.monotonic()
    total_generation_time = end - start

    torch.cuda.synchronize(model.device)
    start = time.monotonic()
    if video_embedding is not None:
        model.generate_with_video_embed(input_ids,
                                        video_embedding,
                                        modalities=["video"],
                                        do_sample=False,
                                        temperature=0,
                                        max_new_tokens=1,
                                        )
    else:
        model.generate(input_ids,
                       do_sample=False,
                       temperature=0,
                       max_new_tokens=1,
                       )
    torch.cuda.synchronize(model.device)
    end = time.monotonic()
    ttft = end - start

    if cont is not None:
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    else:
        text_outputs = ''
        total_generation_time = LLM_INF
        ttft = LLM_INF

    return text_outputs, total_generation_time, ttft


def llava_inference(qs, video):
    if video is not None:
        question = DEFAULT_IMAGE_TOKEN + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )
    if cont is not None:
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    else:
        text_outputs = ''
    return text_outputs


def mm_infer_embedding(video_embeddings, instruct, model, tokenizer, modal='video'):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    if video_embeddings is None:
        modal = 'text'
    else:
        modal = 'video'
    # 1. text preprocess (tag process & generate prompt).

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
                """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                """\n"""
                """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
             }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(
        0).long().to(device)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

    # 3. generate response according to visual signals and prompts.
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = False
    temperature = 0.0
    top_p = 0.9
    max_new_tokens = 2048

    with torch.inference_mode():
        output_ids = model.generate_with_video_embed(
            input_ids,
            attention_mask=attention_masks,
            video_embeddings=video_embeddings,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def llama_inference(qs, video =None):
    input_ids = tokenize_func(video, qs)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

    # 3. generate response according to visual signals and prompts.
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = False
    temperature = 0.0
    top_p = 0.9
    max_new_tokens = 200

    # assert isinstance(model, LlavaQwenForCausalLM)

    cont = model.generate(
        input_ids,
        attention_mask=attention_masks,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        pad_token_id=tokenizer.eos_token_id,
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs

def save_inference_result(llm_name, dataset_name, video_id, questions, modality2config, keyword_extractor):
    result_file_name = get_inference_result_dir(llm_name, dataset_name, video_id, modality2config,
                                                keyword_extractor.name)
    json.dump(questions, open(result_file_name, 'w', encoding='utf-8'))


def load_inference_result(llm_name, dataset_name, video_id, modality2config, keyword_extractor):
    processed_question2results = {}
    result_file_name = get_inference_result_dir(llm_name, dataset_name, video_id, modality2config,
                                                keyword_extractor.name)
    if os.path.exists(result_file_name):
        try:
            results = json.load(open(result_file_name, 'r'))
        except json.decoder.JSONDecodeError:
            os.remove(result_file_name)
            results = []
        for result in results:
            if result.get('pred'):
                processed_question2results[result['question_id']] = result
    return processed_question2results


def check_modality_usage(modality2config):
    return tuple(modality2config.get(modality) for modality in Modality.MainModalities)


def warm_up():
    print('Start warm up')
    input_ids = tokenizer_image_token('Who are you', tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    model.generate(input_ids,
                   do_sample=False,
                   temperature=0,
                   max_new_tokens=10,
                   )
    if inference_parameter.llm_name == LLMName.LlavaVideoQwen7b:
        video = image_processor.preprocess([np.random.random((200, 200, 3)) for _ in range(10)], return_tensors="pt")[
            "pixel_values"].cuda(0).bfloat16()
        video = [video]
        model.image2token(video)
    else:
        video = image_processor.preprocess([np.random.random((200, 200, 3)) for _ in range(10)], return_tensors="pt")[
            "pixel_values"].cuda(0).half()
        video = [(video, 'video')]
        model.image2token(video)
    print('Finish warm up')




def save_retrieve_result(llm_name, retrieve_result, dataset_name, video_id, modality, keyword_extractor, config=None):
    retrieve_result_dir = os.path.join(configuration()['cache_root'], 'legacy', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_retrieve_file_name(modality, config)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    json.dump(retrieve_result, open(retrieve_result_file, 'w'))


def retrieve(llm_name, question, retrieve_result, docs_total, dataset_name, query_key, modality, keyword_extractor,
             config,
             video_id, inference_parameter, json_request=None, json_key=None):
    if retrieve_result.get(query_key) is None or not inference_parameter.use_existing_retrieve_result:
        if modality == Modality.Object:
            global sentence_model
            if sentence_model is None:
                sentence_model, _ = clip.load(configuration()['model_paths']['object_clip'], device=device)
            det_top_idx = [i for i in range(config['num_frame'])]
            retrieve_query = None
            uid2clip = docs_total['uid2clip']
            uid2category = docs_total['uid2category']
            scene_graph_process_time = 1e5
            embed_retrieve_time = 1e5
            for i in range(num_measure_retrieve_latency):
                try:
                    det_retrieve_info = query_json_request(json_request=json_request, key='TYPE')
                    request_det = query_json_request(json_request=json_request, key='DET')
                    retrieve_query = request_det
                except Exception as e:
                    if isinstance(e, KeyError):
                        raise e
                    det_retrieve_info = None

                if retrieve_query is None:
                    if keyword_extractor.name == KeyWordExtractorNames.NonExtractor:
                        retrieve_query = None
                    else:
                        retrieve_query = []
                use_org_label = True if keyword_extractor.name == KeyWordExtractorNames.BareExtractor else False
                retrieved_docs, cur_embed_retrieve_time = get_retrieved_object_result(docs_total[Modality.Object],
                                                                                      retrieve_query,
                                                                                      uid2clip, uid2category,
                                                                                      config, clip.tokenize,
                                                                                      sentence_model, use_org_label)
                embed_retrieve_time = min(embed_retrieve_time, cur_embed_retrieve_time)

                L, R, N = False, False, False

                if det_retrieve_info is not None:
                    if ObjectType.Location in det_retrieve_info:
                        L = True
                    if ObjectType.Relation in det_retrieve_info:
                        R = True
                    if ObjectType.Number in det_retrieve_info:
                        N = True
                start = time.monotonic()
                scene_graph_docs = det_preprocess(retrieved_docs, location=L, relation=R,
                                                  number=N,
                                                  keyword_extractor=keyword_extractor)  # pre-process of APE information
                end = time.monotonic()
                cur_scene_graph_process_time = end - start
                scene_graph_process_time = min(cur_scene_graph_process_time, scene_graph_process_time)
            retrieve_time = scene_graph_process_time + embed_retrieve_time

            retrieve_result[query_key] = {
                'retrieve_time': retrieve_time,
                'result': scene_graph_docs
            }
        else:
            try:
                request_keywords = query_json_request(json_request=json_request, key=json_key)
                if not isinstance(request_keywords, list):
                    request_keywords = [request_keywords]
            except Exception as e:
                if isinstance(e, KeyError):
                    raise e
                request_keywords = None
            retrieve_query = []
            if len(docs_total) > 0:
                if request_keywords is not None and len(request_keywords) > 0:
                    retrieve_query = [request_keywords] if not isinstance(request_keywords, list) else request_keywords
                if isinstance(keyword_extractor, NonKeywordExtractor):
                    retrieve_query = [None]

            faiss_retrieve_time, vectorize_query_time = 1e5, 1e5 if len(retrieve_query) > 0 else 0.0
            for i in range(num_measure_retrieve_latency):
                retrieved_docs, _, cur_faiss_retrieve_time, cur_vectorize_query_time = retrieve_documents_with_dynamic(
                    docs_total, retrieve_query,
                    threshold=config.get('retrieve_threshold', rag_threshold))
                faiss_retrieve_time = min(cur_faiss_retrieve_time, faiss_retrieve_time)
                vectorize_query_time = min(cur_vectorize_query_time, vectorize_query_time)

            retrieve_time = faiss_retrieve_time + vectorize_query_time

            retrieve_result[query_key] = {
                'retrieve_time': retrieve_time,
                'result': retrieved_docs,
                'faiss_retrieve_time': faiss_retrieve_time,
                'vectorize_query_time': vectorize_query_time,
            }

        if inference_parameter.need_save_retrieve_result:
            save_retrieve_result(llm_name, retrieve_result, dataset_name, video_id, modality, keyword_extractor, config)
    else:
        retrieved_docs = retrieve_result[query_key]['result']
        retrieve_time = retrieve_result[query_key]['retrieve_time']
    retrieve_query = query_key
    if modality == Modality.Object:
        if 'scene_graph_docs' not in locals():
            scene_graph_docs = retrieved_docs
            det_top_idx = list(range(config['num_frame']))
            scene_graph_process_time = 0.0
        return scene_graph_docs, det_top_idx, retrieve_time, scene_graph_process_time
    else:
        return retrieved_docs, retrieve_time


def query_json_request(json_request, key):
    request = json_request
    if isinstance(request, str):
        request = request.strip()
        if request.startswith('```'):
            request = '\n'.join(request.splitlines()[1:-1])
        try:
            request = json.loads(request)
        except json.JSONDecodeError:
            request = ast.literal_eval(request)
    if not isinstance(request, dict):
        raise ValueError('Keyword extraction must return a dictionary')
    keys = key if isinstance(key, list) else [key]
    aliases = {'DET': ['DET', 'object', 'OBJ'], 'TYPE': ['TYPE', 'Type', 'type'],
               'OCR': ['OCR', 'ocr'], 'ASR': ['ASR', 'asr']}
    for candidate in keys:
        for alias in aliases.get(candidate, [candidate]):
            if alias in request:
                return request[alias]
    return None


def get_modality_mask(keyword_result, extractor):
    # The selected configuration, rather than the presence of keyword keys,
    # determines which modalities are included. Empty retrieval simply yields no text.
    return {modality: 0 for modality in Modality.MainModalities}


rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0


def run_inference_for_dataset(**run_kwargs):
    video_id2questions = run_kwargs['video_id2questions']
    dataset_path = run_kwargs['dataset_path']
    dataset_name = run_kwargs['dataset_name']
    question_type = run_kwargs['question_type']
    video_id2name = run_kwargs['video_id2name']
    valid_modality2config = run_kwargs['valid_modality2config']
    all_returned = []
    inference_parameter = run_kwargs['inference_parameter']
    llm_name = inference_parameter.llm_name
    keyword_extractor = KeywordExtractorBuilder.build(run_kwargs['keyword_extractor_name'], **{
        'llm_gen_func': llava_inference if llm_name == LLMName.LlavaVideoQwen7b else llama_inference,
        'prompt_factory': pf,
        'llm_name': llm_name
    })
    run_kwargs['keyword_extractor'] = keyword_extractor
    keyword_extractor.warm_up()

    for video_id in list(video_id2questions.keys()):
        run_kwargs['video_id'] = video_id
        run_inference_for_one_video(**run_kwargs)
    inference_parameter.use_existing_keyword_result = True
    inference_parameter.need_save_keyword_result = False

def llama_inference_splited(qs, video_embedding=None):
    input_ids = tokenize_func(video_embedding, qs)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = False
    temperature = 0.0
    top_p = 0.9
    max_new_tokens = 2048


    torch.cuda.synchronize(model.device)
    start = time.monotonic()
    if video_embedding is not None:
        cont = model.generate_with_video_embed(
            input_ids,
            attention_mask=attention_masks,
            video_embeddings=video_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        cont = model.generate(input_ids,
                              attention_mask=attention_masks,
                              video_embeddings=video_embedding,
                              do_sample=do_sample,
                              temperature=temperature,
                              max_new_tokens=max_new_tokens,
                              top_p=top_p,
                              use_cache=True,
                              stopping_criteria=[stopping_criteria],
                              pad_token_id=tokenizer.eos_token_id,
                              )
    torch.cuda.synchronize(model.device)
    end = time.monotonic()
    total_generation_time = end - start

    torch.cuda.synchronize(model.device)
    start = time.monotonic()
    if video_embedding is not None:
        model.generate_with_video_embed(input_ids,
            attention_mask=attention_masks,
            video_embeddings=video_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=1,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
                                        )
    else:
        model.generate(input_ids,
            attention_mask=attention_masks,
            video_embeddings=video_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=1,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
                       )
    torch.cuda.synchronize(model.device)
    end = time.monotonic()
    ttft = end - start

    if cont is not None:
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    else:
        text_outputs = ''
        total_generation_time = LLM_INF
        ttft = LLM_INF

    return text_outputs, total_generation_time, ttft
    pass


def run_inference_for_one_video(**run_kwargs):
    video_id2questions = run_kwargs['video_id2questions']
    dataset_name = run_kwargs['dataset_name']
    question_type = run_kwargs['question_type']
    video_id2name = run_kwargs['video_id2name']
    video_id = run_kwargs['video_id']
    inference_parameter = run_kwargs['inference_parameter']
    llm_name = run_kwargs['llm_name']
    assert isinstance(inference_parameter, InferenceParameter)
    valid_modality2config = run_kwargs['valid_modality2config']
    all_returned = []
    if run_kwargs.get('keyword_extractor'):
        keyword_extractor = run_kwargs['keyword_extractor']
    else:
        keyword_extractor = KeywordExtractorBuilder.build(run_kwargs['keyword_extractor_name'], **{
            'llm_gen_func': llava_inference if llm_name == LLMName.LlavaVideoQwen7b else llama_inference,
            'prompt_factory': pf,
            'llm_name': llm_name
        })
        keyword_extractor.warm_up()
    video_name = video_id2name[video_id]
    keyword_result = load_keyword_result(llm_name, dataset_name, video_id,
                                         keyword_extractor)
    for modality2config_id, modality2config in enumerate(valid_modality2config):

        visiontoken_config, caption_config, object_config, ocr_config, asr_config = check_modality_usage(
            modality2config)
        if asr_config is not None:
            asr_retrieve_result = load_retrieve_result(llm_name, dataset_name, video_id, Modality.ASR,
                                                       keyword_extractor,
                                                       asr_config)
        if ocr_config is not None:
            ocr_retrieve_result = load_retrieve_result(llm_name, dataset_name, video_id, Modality.OCR,
                                                       keyword_extractor,
                                                       ocr_config)
        if object_config is not None:
            object_retrieve_result = load_retrieve_result(llm_name, dataset_name, video_id, Modality.Object,
                                                          keyword_extractor,
                                                          object_config)
        processed_question2results = load_inference_result(llm_name, dataset_name, video_id, modality2config,
                                                           keyword_extractor)

        requested_ids = run_kwargs.get('question_ids')
        question_dicts = [q for q in video_id2questions[video_id]
                          if requested_ids is None or q['question_id'] in requested_ids]
        inference_results = []
        question_set = set()

        visiontoken_load_time = None
        preprocess_times = {}
        visiontoken = None

        if visiontoken_config is not None:
            visiontoken_path = preprocess_store_path(Modality.VisionToken, visiontoken_config, video_name,
                                                     dataset_name, llm_name)
            start = time.monotonic()
            visiontoken_process_result = pickle.load(open(visiontoken_path, 'rb'))
            visiontoken = visiontoken_process_result[Modality.VisionToken]
            end = time.monotonic()
            visiontoken_load_time = end - start
        if caption_config is not None:
            caption_path = preprocess_store_path(Modality.Caption, caption_config, video_name, dataset_name)
            caption_process_result = pickle.load(open(caption_path, 'rb'))
            caption_docs = caption_process_result[Modality.Caption]
        if object_config is not None:
            object_path = preprocess_store_path(Modality.Object, object_config, video_name, dataset_name)
            object_process_result = pickle.load(open(object_path, 'rb'))
        if ocr_config is not None:
            ocr_path = preprocess_store_path(Modality.OCR, ocr_config, video_name, dataset_name)
            orc_process_result = pickle.load(open(ocr_path, 'rb'))
            ocr_docs_total = orc_process_result[Modality.OCR]
        if asr_config is not None:
            asr_path = preprocess_store_path(Modality.ASR, asr_config, video_name, dataset_name)
            asr_process_result = pickle.load(open(asr_path, 'rb'))
            asr_docs_total = asr_process_result[Modality.ASR]

        for question_dict in question_dicts:
            question = question_dict['question']
            question_id = question_dict['question_id']
            query_key = question_id
            if question_id in question_set:
                raise Exception
            question_set.add(question_id)

            if not inference_parameter.force_inference:
                if question_id in processed_question2results:
                    inference_result = processed_question2results[question_id]
                    inference_results.append(inference_result)
                    continue

            inference_result = copy.deepcopy(question_dict)
            inference_results.append(inference_result)

            asr_retrieve_time = None
            ocr_retrieve_time = None
            object_retrieve_time = None
            online_object_process_time = None
            scene_graph_process_time = None
            caption_load_time = None

            video_meta_path = get_video_meta_path(video_name, dataset_name)
            video_meta = pickle.load(open(video_meta_path, 'rb'))

            start = time.monotonic()
            if not any(m in modality2config for m in Modality.RetrieveModalities):
                json_request = {m: None for m in Modality.RetrieveModalities}
                keyword_extraction_time = 0.0
            elif keyword_result.get(question) is None or not inference_parameter.use_existing_keyword_result:
                retrieve_pmt_0 = pf.retrieve_prompt(question)
                keyword_extraction_time = 1000
                for i in range(num_measure_retrieve_latency):
                    start = time.monotonic()
                    json_request = keyword_extractor.extract_keywords(question, modalities=modality2config.keys())
                    end = time.monotonic()
                    keyword_extraction_time = min(keyword_extraction_time, end - start)
                keyword_result[question] = {
                    'retrieve_time': keyword_extraction_time,
                    'result': json_request
                }
                if inference_parameter.need_save_keyword_result:
                    save_keyword_result(llm_name, keyword_result, dataset_name, video_id, keyword_extractor)
            else:
                json_request, keyword_extraction_time = keyword_result[question]['result'], \
                    keyword_result[question]['retrieve_time']
            end = time.monotonic()

            modality_mask = get_modality_mask(json_request, keyword_extractor)
            object_docs = None
            if object_config is not None and not modality_mask[Modality.Object]:
                object_docs, det_top_idx, object_retrieve_time, scene_graph_process_time = retrieve(llm_name, question,
                                                                                                    object_retrieve_result,
                                                                                                    object_process_result,
                                                                                                    dataset_name,
                                                                                                    query_key,
                                                                                                    Modality.Object,
                                                                                                    keyword_extractor,
                                                                                                    config=object_config,
                                                                                                    inference_parameter=inference_parameter,
                                                                                                    video_id=video_id,
                                                                                                    json_request=json_request,
                                                                                                    json_key='DET')

                assert object_retrieve_time >= scene_graph_process_time
            # OCR fetch
            start = time.monotonic()
            ocr_docs = None
            if ocr_config is not None and not modality_mask[Modality.OCR]:
                ocr_docs = []
                if len(ocr_docs_total) > 0:
                    ocr_docs, ocr_retrieve_time = retrieve(llm_name, question, ocr_retrieve_result, ocr_docs_total,
                                                           dataset_name,
                                                           query_key, Modality.OCR, keyword_extractor,
                                                           config=ocr_config, video_id=video_id,
                                                           inference_parameter=inference_parameter,
                                                           json_request=json_request, json_key=['OCR', 'ocr'])

                end = time.monotonic()
            # ASR fetch
            start = time.monotonic()
            asr_docs = None
            if asr_config is not None and not modality_mask[Modality.ASR]:
                asr_docs = []
                if len(asr_docs_total) > 0:
                    asr_docs, asr_retrieve_time = retrieve(llm_name, question, asr_retrieve_result, asr_docs_total,
                                                           dataset_name,
                                                           query_key, Modality.ASR, keyword_extractor,
                                                           config=asr_config, video_id=video_id,
                                                           inference_parameter=inference_parameter,
                                                           json_request=json_request, json_key=['ASR', 'asr'])

                end = time.monotonic()
            modality2doc = {}
            if visiontoken is not None and not modality_mask[Modality.VisionToken]:
                modality2doc[Modality.VisionToken] = None
            if object_config and object_docs is not None and len(object_docs) > 0:
                modality2doc[Modality.Object] = {
                    'det_docs': object_docs,
                    'det_top_idx': det_top_idx,
                    'max_frames_num': object_config['num_frame']
                }
            if asr_config and asr_docs is not None and len(asr_docs) > 0:
                modality2doc[Modality.ASR] = asr_docs
            if ocr_config and ocr_docs is not None and len(ocr_docs) > 0:
                modality2doc[Modality.OCR] = ocr_docs

            if caption_config and len(caption_docs) > 0:
                modality2doc[Modality.Caption] = caption_docs

            final_prompt = pf.get_prompt(modality2doc=modality2doc, modality2config=modality2config,
                                         question_type=question_type,
                                         question_dict=question_dict, video_meta=video_meta,
                                         dataset_name=dataset_name, **{
                    'tokenize_func': tokenize_func,
                    'tokenizer': tokenizer,
                    'IMAGE_TOKEN_INDEX': IMAGE_TOKEN_INDEX,
                    'video_embedding': visiontoken[0] if visiontoken is not None else None,
                    'max_context_length': max_context_length,
                    'token_margin': token_margin
                })
            try:
                assert question.split('\n')[-1] in final_prompt
            except AssertionError as e:
                raise e
            if inference_parameter.do_inference:
                if llm_name == LLMName.LlavaVideoQwen7b:
                    answer, generation_time, ttft = llava_inference_splited(final_prompt, visiontoken)
                elif llm_name == LLMName.VideoLlamaMistral7b:
                    answer, generation_time, ttft = llama_inference_splited(final_prompt, visiontoken)
                else:
                    raise NotImplementedError
                inference_result['pred'] = answer
                inference_result['generation_time'] = generation_time
                inference_result['ttft'] = ttft
                ids = tokenize_func(visiontoken, final_prompt)
                if visiontoken is not None:
                    inference_result['token_count'] = int(prepare_cached_inputs(model, ids, visiontoken).shape[1])
                    blocks = visiontoken if isinstance(visiontoken, (list, tuple)) else [visiontoken]
                    inference_result['modality_tokens'] = {'visiontoken': sum(int(b.shape[-2]) for b in blocks)}
                else:
                    inference_result['token_count'] = int(ids.shape[1])
                    inference_result['modality_tokens'] = {}
                inference_result['keyword_extraction_time'] = keyword_extraction_time
                inference_result['asr_retrieve_time'] = asr_retrieve_time
                inference_result['ocr_retrieve_time'] = ocr_retrieve_time
                inference_result['object_retrieve_time'] = object_retrieve_time
                inference_result['visiontoken_load_time'] = visiontoken_load_time
                inference_result['caption_load_time'] = caption_load_time
                inference_result['prompt'] = final_prompt
                if inference_parameter.need_save_inference_result:
                    save_inference_result(llm_name, dataset_name, video_id, inference_results, modality2config,
                                          keyword_extractor)
                print(
                    f'Dataset {dataset_name} video_id: {video_id} question: {question}  Config: {modality2config_id}:{modality2config} Answer: {answer}')

        if inference_parameter.need_save_inference_result:
            save_inference_result(llm_name, dataset_name, video_id, inference_results, modality2config,
                                  keyword_extractor)
        all_returned.extend(inference_results)
    return all_returned


inference_parameter = InferenceParameter()
inference_parameter.use_existing_retrieve_result = True
inference_parameter.use_existing_keyword_result = True
inference_parameter.do_inference = True
inference_parameter.need_save_inference_result = True
inference_parameter.need_save_retrieve_result = True
inference_parameter.need_save_keyword_result = True
inference_parameter.force_inference = False

num_measure_retrieve_latency = 3
