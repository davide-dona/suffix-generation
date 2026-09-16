import json
from contextlib import ExitStack
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.artifacts import sha256
from src.runtime import output_path, start_stage
from src.visualization.labels import MODELS
from src.visualization.suffix_drawing import render
from src.visualization.suffix_graphs import Artifact, Criteria, align, shortlist


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def _open(paths: list[Path], stack: ExitStack) -> list[Artifact]:
    artifacts = []
    for path in paths:
        artifact = Artifact(path)
        stack.callback(artifact.close)
        artifacts.append(artifact)
    order = MODELS.ordered(artifact.model for artifact in artifacts)
    return sorted(artifacts, key=lambda artifact: order.index(artifact.model))


def make_shortlist(paths: list[Path], criteria: Criteria, threshold: float, destination: Path) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError('label_threshold must be between zero and one')
    with ExitStack() as stack:
        artifacts = _open(paths, stack)
        candidates, report = shortlist(artifacts, criteria)
        sources = [artifact.provenance() for artifact in artifacts]
        report['sources'] = sources
        report['candidates'] = candidates
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / 'shortlist.json', report)
        if report['fewer_than_requested']:
            print(f'Only {len(candidates)} prefix groups qualified; criteria were not relaxed.')
        print(f'Matched keys: {report["matched_keys"]}; unmatched: {report["unmatched_keys"]}')
        for candidate in candidates:
            key = (candidate['case_id'], candidate['prefix_len'])
            row = artifacts[0].rows[key]
            selection = {
                'version': 1,
                'status': 'candidate',
                'rationale': '',
                'dataset': artifacts[0].dataset,
                'case_id': key[0], 'prefix_len': key[1],
                'prefix_activities': row['prefix_activities'],
                'true_activities': row['true_activities'],
                'sources': sources,
                'criteria': report['criteria'],
                'candidate': candidate,
                'label_threshold': threshold,
                'provenance_limitations': [
                    'Generation artifacts do not store complete conditioning inputs or '
                    'termination reasons; matching verifies case, cut, activities, and observation.',
                    *[
                        f'{artifact.model}: checkpoint hash unavailable in legacy provenance.'
                        for artifact in artifacts
                        if artifact.metadata['checkpoint_sha256'].startswith('legacy-unavailable')
                    ],
                ],
            }
            directory = destination / f'candidate-{candidate["rank"]:02d}'
            graphs = {artifact.model: artifact.graphs([key])[key] for artifact in artifacts}
            print(f'Rendering candidate {candidate["rank"]}: {key}', flush=True)
            render(selection, graphs, artifacts[0].vocabulary, directory)
            _write_json(directory / 'selection.json', selection)


def render_selection(path: Path, destination: Path) -> None:
    selection = json.loads(path.read_text())
    if selection.get('version') != 1:
        raise ValueError('Unsupported selection version')
    if selection.get('status') not in ('candidate', 'selected'):
        raise ValueError('Selection status must be candidate or selected')
    if selection['status'] == 'selected' and not selection.get('rationale', '').strip():
        raise ValueError('A selected example must record a selection rationale')
    paths = []
    for source in selection['sources']:
        artifact_path = Path(source['path'])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if sha256(artifact_path) != source['sha256']:
            raise ValueError(f'Generation artifact hash mismatch: {artifact_path}')
        paths.append(artifact_path)
    with ExitStack() as stack:
        artifacts = _open(paths, stack)
        align(artifacts)
        if artifacts[0].dataset != selection['dataset']:
            raise ValueError('Selection dataset does not match its artifacts')
        key = (selection['case_id'], selection['prefix_len'])
        if any(key not in artifact.rows for artifact in artifacts):
            raise ValueError(f'Selected case/cut is missing: {key}')
        row = artifacts[0].rows[key]
        if any(selection[field] != row[field] for field in ('prefix_activities', 'true_activities')):
            raise ValueError('Selection prefix or observation does not match its artifacts')
        graphs = {artifact.model: artifact.graphs([key])[key] for artifact in artifacts}
        render(selection, graphs, artifacts[0].vocabulary, destination)
        _write_json(destination / 'selection.json', selection)


@hydra.main(version_base='1.3', config_path='../config', config_name='compare_suffixes')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    destination = output_path('comparison')
    if cfg.mode == 'shortlist':
        if cfg.selection or not cfg.generations:
            raise ValueError('Shortlist mode requires generations and no selection')
        make_shortlist(
            [Path(path) for path in cfg.generations],
            Criteria(**OmegaConf.to_container(cfg.shortlist)),
            float(cfg.label_threshold), destination,
        )
    elif cfg.mode == 'render':
        if not cfg.selection or cfg.generations:
            raise ValueError('Render mode requires selection and no generations')
        render_selection(Path(cfg.selection), destination)
    else:
        raise ValueError('mode must be shortlist or render')
    print(f'Wrote suffix comparisons to {destination}')


if __name__ == '__main__':
    main()
