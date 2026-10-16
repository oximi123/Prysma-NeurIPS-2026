# Prysma: Efficient Modality Adaptation for SLO-aware LLM-based Video Question Answering

Prysma adapts video modalities and extraction parameters to improve answer quality under a Time-to-First-Token (TTFT) service-level objective. It combines an offline modality orchestrator with an online semantic adapter.

The offline orchestrator builds a video knowledge base from CLIP features, uses a latency-aware pruner to filter configurations, and performs two-stage tuning with TPE and a genetic algorithm. The online adapter selects a configuration for each incoming question using semantic similarity and the video's tuning history.

The pipeline supports vision tokens, captions, object detection, OCR, and audio transcripts, with LLaVA-Video-7B and VideoLLaMA2-7B as video backbones. It also includes VT and Random comparison methods, together with VQA adaptations of [SCOOT](https://github.com/ant-research/SCOOT-SLO-Oriented-Performance-Tuning), [LLMConf](https://github.com/IntelligentDDS/LLMConf), and [OPPerTune](https://github.com/microsoft/OPPerTune). These adaptations implement the referenced optimization mechanisms for the Prysma pipeline; their official implementations are maintained in the linked repositories.

## 1. Install dependencies

Clone the repository:

```bash
git clone https://github.com/oximi123/Prysma-NeurIPS-2026.git
cd Prysma-NeurIPS-2026
```

Use Python 3.10 or 3.11. Install FFmpeg and a C/C++ build toolchain before installing the video dependencies. Run the following commands from the project directory:

```bash
python -m pip install --upgrade pip setuptools wheel Cython
python -m pip install -r requirements.txt
```

Install the upstream [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2), [LaViLa](https://github.com/facebookresearch/LaViLa), [APE](https://github.com/shenyunhang/APE), and [Katna](https://github.com/keplerlab/katna) dependencies. Install the `llava`, `videollama2`, `ape`, `detectron2`, and `Katna` packages in the active environment, and place the LaViLa sources under `preprocess_servers/LaViLa`. Match the upstream packages to the checkpoints you configure.

Download the checkpoints used by the configured modalities, including the video backbones, caption models, CLIP, Whisper, RTDETR, APE, and DINOv2. Set their paths in the configuration before starting the extraction services.

## 2. Configure paths and services

Edit `configs/main_result.json`. The main runtime settings are:

| Setting | Description |
| --- | --- |
| `runtime.data_root` | Dataset directory |
| `runtime.preprocess_root` | Extracted modality storage |
| `runtime.cache_root` | Inference and scoring cache |
| `runtime.output_root` | Tuning and evaluation outputs |
| `runtime.model_paths` | Local checkpoint and model configuration paths |
| `runtime.device` | Inference device, such as `cuda:0` |
| `runtime.server_host` / `runtime.server_ports` | Extraction service addresses |
| `runtime.judge` | Scoring model, API endpoint, and API-key environment variable |
| `runtime.serving_context` | Hardware and serving settings for latency calibration |
| `runtime.latency_profiles` | Paths to the calibrated latency profiles |

Relative paths are resolved against the project directory. The default scoring model is `gpt-3.5-turbo-0125`; configure the model and endpoint you intend to use.

Set the API key in the shell that runs the pipeline:

```bash
export OPENAI_API_KEY="your-api-key"
```

In PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

To use another configuration, pass `--config /absolute/path/config.json` to the pipeline commands. Set the same configuration for each extraction service process:

```bash
export PRYSMA_CONFIG="/absolute/path/config.json"
```

In PowerShell:

```powershell
$env:PRYSMA_CONFIG = "C:\absolute\path\config.json"
```

## 3. Prepare datasets

Place videos under `data/{dataset}/videos`. The annotation files are:

| Dataset | Annotation files |
| --- | --- |
| MSVD-QA | `train.jsonl`, `val.jsonl`, `test.jsonl`, or `train_qa.json`, `val_qa.json`, `test_qa.json` |
| MSRVTT-QA | `train_qa.json`, `val_qa.json`, `test_qa.json` |
| EgoTimeQA | `annotations.EgoTimeQA.json` |

JSON files contain a list of records with `video_id`, `question`, and `answer` (or `a`). JSONL files contain one record per line. For EgoTimeQA, the loader also accepts an existing `GroundQA` directory.

For example:

```text
data/
  MSVD-QA/
    videos/
    train.jsonl
    val.jsonl
    test.jsonl
  MSRVTT-QA/
    videos/
    train_qa.json
    val_qa.json
    test_qa.json
  EgoTimeQA/
    videos/
    annotations.EgoTimeQA.json
```

The `datasets` section of the configuration controls the video limit and tuning-query fraction for each dataset.

## 4. Start extraction services

Start each service in a separate terminal from the project directory:

```bash
python -m preprocess_servers.frame_server
python -m preprocess_servers.visiontoken_server
python -m preprocess_servers.caption_server
python -m preprocess_servers.object_server
python -m preprocess_servers.ocr_server
python -m preprocess_servers.asr_server
python -m preprocess_servers.clip_server
```

Use the shared device settings and each process's GPU environment to allocate the services to your hardware.

Extract video features:

```bash
python script_preprocess.py --dataset MSVD-QA --model LLaVA-Video-7B
```

To pre-extract the modality parameter grids as well:

```bash
python script_preprocess.py --dataset MSVD-QA --model LLaVA-Video-7B --all-modalities
```

The pipeline stores extracted artifacts under `runtime.preprocess_root` and reuses them during tuning. Missing artifacts are extracted on demand by the corresponding service.

## 5. Calibrate latency

Collect token-count and TTFT measurements for each backbone:

```bash
python -m experiments.profile_latency --model LLaVA-Video-7B --dataset MSRVTT-QA --measurements profiles/llava_measurements.jsonl --collect
python -m experiments.profile_latency --model VideoLLaMA2-7B --dataset MSRVTT-QA --measurements profiles/llama_measurements.jsonl --collect
```

These commands save profiles to the paths in `runtime.latency_profiles`. Use `--videos` and `--repeats` to adjust the calibration sample size. To build a profile from existing measurement records, run the same command without `--collect`.

Use profiles calibrated for the model, hardware, and serving settings specified in `runtime.serving_context`. Recalibrate when those settings change.

## 6. Run Prysma

Run the configured combinations of models, datasets, SLOs, methods, and random seeds:

```bash
python main.py
```

Select a particular setting with command-line arguments:

```bash
python main.py --model LLaVA-Video-7B --dataset MSVD-QA --slo 0.5 --seed 42 --method Prysma
```

List the configured settings before starting:

```bash
python main.py --list
```

The top-level configuration controls `models`, `datasets`, `slos`, `methods`, `seeds`, `budget`, and `knowledge_base_fraction`. The runner saves tuning histories and predictions, and resumes completed videos on subsequent runs. Use a separate output directory when changing the configuration or dataset selection.

## Citation

```bibtex
@inproceedings{zhu2026prysma,
  title={Prysma: Efficient Modality Adaptation for SLO-aware LLM-based Video Question Answering},
  author={Zhu, Botao and Fan, Xiaoyi and Zhu, Yifei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
