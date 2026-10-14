import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def pack(best, archive):
    best = Path(best)
    verification = json.loads((best / 'verification.json').read_text())
    assert verification['val_examples'] == 103500
    assert verification['submission_rows'] == 172500
    assert verification['submission_differences'] == 0
    files = [Path(name) for name in ['train.py', 'improve.py', 'ensemble.py', 'verify.py',
                                    'pack.py', 'requirements.txt', 'README.md',
                                    'quick_draw.ipynb', 'results.ipynb']]
    if best.name.startswith('vit-'):
        files.extend([Path('vit.py'), Path('vit_tta.py'), Path('class_errors.py'), Path('vit_results.ipynb')])
    if best.name.startswith('pretrained-vit-'):
        files.extend([Path('pretrained_vit.py'), Path('finish_pretrained.py'), Path('pretrained_results.ipynb')])
    files.append(Path('input/baseline.pth'))
    folders = [Path('output') / name for name in ['run-20261014', 'resize56-20261014',
                                               'regularized-20261014', 'regularized-fixed-20261014', 'ensemble-20261014']]
    if best not in folders:
        folders.append(best)
    names = {'config.json', 'summary.json', 'verification.json', 'history.csv', 'scores.csv',
             'submission.csv', 'train.log', 'environment.txt', 'source.py', 'train_source.py',
             'resnet18_3.pth', 'resnet18_7.pth', 'finetuned.pth', 'model.pth', 'calibration.csv',
             'vit.pth', 'best.json'}
    for folder in folders:
        if (folder / 'summary.json').exists():
            files.extend(p for p in folder.iterdir() if p.name in names)
    records = []
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        records.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    manifest = {'best_output': str(best), 'verification': verification, 'files': records}
    archive = Path(archive)
    prefix = archive.stem
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in files:
            bundle.write(path, str(Path(prefix) / path))
        bundle.writestr(f'{prefix}/manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'files': len(files), 'best_accuracy': verification['accuracy']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('best')
    parser.add_argument('--archive', default='Карсалиев_Тимур_HW_2.zip')
    args = parser.parse_args()
    pack(args.best, args.archive)
