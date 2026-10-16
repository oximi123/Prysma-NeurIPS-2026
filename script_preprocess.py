"""Pre-extract CLIP features, optionally every per-modality knob combination."""
import argparse
import itertools
import os
from pathlib import Path
from config_tuner.records import Configuration
from config_tuner.space import default_space
from experiments.backend import VideoBackend
from experiments.data import load_videos
from experiments.run_main_result import load_settings
from util.runtime_config import DEFAULT_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model', nargs='+')
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--all-modalities', action='store_true', help='Extract each modality grid independently, not all joint MCs')
    args = parser.parse_args(argv)
    os.environ['PRYSMA_CONFIG'] = str(args.config.resolve())
    settings = load_settings(args.config)
    for dataset in args.dataset or settings['datasets']:
        videos = load_videos(settings['runtime']['data_root'], dataset, settings['datasets'][dataset]['video_limit'])
        space = default_space(dataset)
        for model in args.model or settings['models']:
            for video in videos:
                backend = VideoBackend(settings['runtime'], model, dataset, video, 'preparation')
                backend.feature()
                if args.all_modalities:
                    for modality, knobs in space.modalities.items():
                        names = list(knobs)
                        for values in itertools.product(*(knobs[k] for k in names)):
                            cfg = space.normalize(Configuration({modality: dict(zip(names, values))}))
                            backend.prepare(cfg)
                print(f'{model} {dataset} {video.video_id}: preprocessing complete', flush=True)


if __name__ == '__main__':
    main()
