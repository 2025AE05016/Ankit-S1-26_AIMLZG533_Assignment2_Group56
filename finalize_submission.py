"""Validate executed deliverables and build one group submission archive."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from udl.pipeline import load_config
from udl.reporting import build_report, report_ready

NOTEBOOKS = ['UDL_Group56_PartA.ipynb', 'UDL_Group56_PartB.ipynb']

def main():
    cfg = load_config(require_training_device=False); observations = json.loads(Path('observations.json').read_text())
    report_ready(cfg, observations)
    for name in NOTEBOOKS:
        nb = json.loads(Path(name).read_text())
        required = [cell for cell in nb['cells'] if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('required_output')]
        if not required: raise RuntimeError(f'No required-output cells in {name}')
        for cell in required:
            if cell.get('execution_count') is None or not cell.get('outputs'):
                raise RuntimeError(f'{name}: run all required cells and SAVE before finalizing.')
        for cell in nb['cells']:
            if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
                raise RuntimeError(f'{name} contains an error output. Resolve it and rerun.')
        subprocess.run([sys.executable, '-m', 'nbconvert', '--to', 'html', name], check=True)
    report = build_report(cfg, final=True)
    target = Path('UDL_Group56_Assignment2_Submission.zip')
    files = [Path(n) for n in NOTEBOOKS] + [Path(n).with_suffix('.html') for n in NOTEBOOKS]
    files += [report, Path('BITS_lab_usage.jpg'), Path('config.json'), Path('observations.json'),
              Path('requirements.txt'), Path('run.py'), Path('README.md'), Path('finalize_submission.py'),
              Path('test_implementation.py'), Path('test_pipeline.py'), Path('test_reporting.py')]
    files += sorted(Path('udl').glob('*.py'))
    root = Path(cfg['run_dir'])
    files += sorted(root.rglob('*.json')) + sorted(root.rglob('*.png'))
    # Dataset, large training checkpoints, local test artifacts and smoke outputs are intentionally excluded.
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.as_posix())
    print('Ready for group review:', target)
    print('Review the PDF and JPG yourself before uploading one submission for Group 56.')

if __name__ == '__main__': main()
