"""Export genuine records only; missing videos/seeds never become zero scores."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev
from experiments.run_main_result import atomic_json, context_path


def export_results(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    settings = manifest['settings']
    seed_rows, summary_rows = [], []
    with (root / 'per_question.jsonl').open('w', encoding='utf-8') as output:
        for model, dataset, slo, method in itertools.product(settings['models'], settings['datasets'],
                                                            settings['slos'], settings['methods']):
            complete_scores = []
            for seed in settings['seeds']:
                videos = manifest['datasets'].get(dataset, {}).get('videos', {})
                scores, missing, count = [], [], 0
                for vid, expected_ids in videos.items():
                    path = context_path(root, model, dataset, slo, seed) / method / 'videos' / (vid + '.json')
                    if not path.is_file():
                        missing.append(vid)
                        continue
                    data = json.loads(path.read_text(encoding='utf-8'))
                    context = data['context']
                    if (context['model'], context['dataset'], context['slo'], context['seed'], data['method']) != (model, dataset, slo, seed, method):
                        raise ValueError('Result group/context mismatch')
                    if context['fingerprint'] != manifest['fingerprint'] or context['qa_digest'] != manifest['datasets'][dataset]['qa_digest']:
                        raise ValueError('Result configuration/dataset fingerprint mismatch')
                    actual = [p['question_id'] for p in data['predictions']]
                    if not data.get('complete') or len(actual) != len(expected_ids) or set(actual) != set(expected_ids):
                        missing.append(vid)
                        continue
                    for row in data['predictions']:
                        if not math.isfinite(row['raw_score']) or not 0 <= row['raw_score'] <= 5 or not math.isfinite(row['ttft']) or row['ttft'] < 0:
                            raise ValueError('Invalid raw score or TTFT in evaluation records')
                        quality = row['raw_score'] if row['ttft'] <= slo else 0.0
                        if abs(quality - row['quality']) > 1e-10:
                            raise ValueError('Stored quality does not use per-request SLO penalties')
                        scores.append(quality)
                        output.write(json.dumps({**row, 'model': model, 'dataset': dataset, 'slo': slo,
                                                 'seed': seed, 'method': method, 'video_id': vid}, ensure_ascii=False) + '\n')
                    count += 1
                complete = bool(videos) and not missing and bool(scores)
                score = mean(scores) if complete else None
                seed_rows.append({'model': model, 'dataset': dataset, 'slo': slo, 'method': method, 'seed': seed,
                                  'quality': score, 'complete': complete, 'videos': count, 'questions': len(scores),
                                  'missing_videos': len(missing), 'dataset_manifest_missing': not bool(videos)})
                if complete:
                    complete_scores.append(score)
            complete = len(complete_scores) == len(settings['seeds'])
            summary_rows.append({'model': model, 'dataset': dataset, 'slo': slo, 'method': method,
                                 'mean': mean(complete_scores) if complete else None,
                                 'std': stdev(complete_scores) if complete and len(complete_scores) > 1 else None,
                                 'completed_seeds': len(complete_scores), 'expected_seeds': len(settings['seeds']),
                                 'complete': complete})
    for name, rows in [('per_seed.csv', seed_rows), ('fig5_summary.csv', summary_rows)]:
        with (root / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(root / 'export_status.json', {'complete': all(r['complete'] for r in summary_rows),
                'groups': len(summary_rows), 'complete_groups': sum(r['complete'] for r in summary_rows),
                'aggregation': 'question-weighted mean per seed; sample standard deviation across seeds (ddof=1)'})
    return seed_rows, summary_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export Fig. 5 quality means and error bars')
    parser.add_argument('results', type=Path)
    args = parser.parse_args(argv)
    _, rows = export_results(args.results)
    print(f"Exported {len(rows)} groups; {sum(r['complete'] for r in rows)} complete")


if __name__ == '__main__':
    main()
