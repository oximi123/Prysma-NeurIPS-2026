from util.runtime_config import configuration, model_path, ROOT

import torch

from util.preprocess_config import PrepConfig
from util.util import server_config, preprocess_store_path, get_video_path
import numpy as np
import socket
import pickle
import os
import time





modality = 'clip'
overwrite_config = {}


def check_config_consistency(frame_config, ocr_config):
    assert frame_config['num_frame'] == ocr_config['num_frame']
    assert frame_config['sampling_method'] == ocr_config['sampling_method']


from transformers import CLIPProcessor, CLIPModel
clip_model = clip_processor = None

def load_clip_model():
    global clip_model, clip_processor
    if clip_model is None:
        checkpoint = configuration()['feature_model']
        clip_model = CLIPModel.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                               device_map=configuration()['device'])
        clip_processor = CLIPProcessor.from_pretrained(checkpoint)
        clip_model.eval()

def clip_process(frames, config):
    load_clip_model()
    start_time = time.time()
    video_tensor = []
    for frame in frames:
        processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device,
                                                                                         dtype=torch.float16)
        video_tensor.append(processed.squeeze(0))
    video_tensor = torch.stack(video_tensor, dim=0)
    with torch.no_grad():
        clip_img_feats = clip_model.get_image_features(video_tensor).float().cpu().numpy()
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        'clip': clip_img_feats,
        "processing_time": processing_time,
        "config": config,
    }


def warm_up():
    print('Start warm up with a sample video')
    clip_process([np.random.random((200, 200, 3)) for _ in range(10)], PrepConfig.clip_default_config())
    print('Finish warm up with a sample video')


force_overwrite = False
from omegaconf import OmegaConf


def main():
    # Set up a socket server
    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((configuration()['server_host'], port))
    server_socket.listen()

    print(modality + " server is listening...")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")

            data = client_socket.recv(8192)

            # Deserialize datas

            dataset_name, video_name, config = pickle.loads(data)
            if dataset_name is None and video_name is None and config is None:
                print(f'{modality} Server closed!')
                break

            video_path = get_video_path(dataset_name, video_name)
            assert config['modality'] == modality

            frame_path = preprocess_store_path('frame', config, video_name, dataset_name)
            store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name)

            if not os.path.exists(store_path) or force_overwrite:
                frame_result = pickle.load(open(frame_path, 'rb'))
                check_config_consistency(frame_result['config'], config)

                frames = frame_result['frames']
                result = clip_process(frames, config)

                pickle.dump(result, open(store_path, 'wb'))
                del result
            # Send back result
            client_socket.send(pickle.dumps(store_path))
            client_socket.close()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
