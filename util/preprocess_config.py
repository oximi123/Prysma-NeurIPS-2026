"""Default configurations for extraction-service warm-up helpers."""
from omegaconf import OmegaConf


class PrepConfig:
    @staticmethod
    def frame_default_config():
        return OmegaConf.create({'modality': 'frame', 'num_frame': 16, 'sampling_method': 'uniform'})

    @staticmethod
    def visiontoken_default_config():
        return OmegaConf.create({'modality': 'visiontoken', 'num_frame': 16, 'sampling_method': 'uniform'})

    @staticmethod
    def clip_default_config():
        return OmegaConf.create({'modality': 'clip', 'num_frame': 16, 'sampling_method': 'uniform'})

    @staticmethod
    def caption_default_config():
        return OmegaConf.create({'modality': 'caption', 'seconds_per_caption': 2,
                                 'frames_per_caption': 4, 'num_return_sequences': 5, 'temperature': 0.7})

    @staticmethod
    def asr_default_config():
        return OmegaConf.create({'modality': 'asr', 'model': 'whisper-large', 'chunk_length': 3, 'sampling_rate': 16000})

    @staticmethod
    def ocr_default_config():
        return OmegaConf.create({'modality': 'ocr', 'num_frame': 16, 'sampling_method': 'uniform', 'text_threshold': 0.3})
