import csv
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path


source = Path('output/pretrained-vit-95-20261015')
output = Path('output/pretrained-vit-95-plateau-20261015')
trainer = Path(__file__).with_name('pretrained_vit.py')


def status(stage, **values):
    record = {'stage': stage, 'updated_at': time.time(), **values}
    (output / 'queue.tmp').write_text(json.dumps(record, indent=2))
    (output / 'queue.tmp').replace(output / 'queue.json')
    print(json.dumps(record), flush=True)


def main():
    assert not (output / 'config.json').exists()
    pid = int((source / 'job.pid').read_text())
    try:
        handle = os.pidfd_open(pid)
    except ProcessLookupError:
        handle = None
    if handle is not None:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
        assert b'vit95-job-20261015.sh' in args
        status('waiting', after_pid=pid, extra_epochs=10, schedule='plateau')
        waiter = select.poll()
        waiter.register(handle, select.POLLIN)
        waiter.poll()
        os.close(handle)
    assert (source / 'exit.code').read_text().strip() == '0', 'Первый прогон завершился с ошибкой'
    verified = json.loads((source / 'verification.json').read_text())
    assert verified['val_examples'] == 34500 and verified['submission_rows'] == 172500
    assert verified['submission_differences'] == 0
    rows = list(csv.DictReader((source / 'history.csv').open()))
    assert [int(row['epoch']) for row in rows] == list(range(1, 11))
    checkpoint = source / 'epoch-10.pth'
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    status('training', source_checkpoint=str(checkpoint), sha256=digest, extra_epochs=10,
           schedule='plateau', deadline_unix=time.time() + 10800)
    deadline = json.loads((output / 'queue.json').read_text())['deadline_unix']
    command = [sys.executable, '-u', str(trainer), '--resume', str(checkpoint),
               '--output', str(output), '--epochs', '10', '--schedule', 'plateau',
               '--resume-lr', '0.00001', '--deadline', str(deadline)]
    child = subprocess.Popen(command)
    (output / 'training.pid').write_text(str(child.pid) + '\n')
    return_code = child.wait()
    assert return_code == 0, 'Продолжение обучения завершилось с ошибкой'
    status('verification')
    subprocess.run([sys.executable, '-u', str(trainer), '--verify', '--output', str(output)], check=True)
    notebook = Path(__file__).with_name('vit95_plateau_results.ipynb')
    subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                    '--inplace', '--ExecutePreprocessor.timeout=120', str(notebook)], check=True)
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        (output / 'exit.code').write_text('1\n')
        raise
    (output / 'exit.code').write_text('0\n')
