# Quick, Draw!

`quick_draw.ipynb` — локальный запуск: проверка baseline, сравнение ResNet с шириной 32 и 64, дообучение победителя, `submission.csv`.

`train.py` выполняет тот же код без Jupyter. `quick_draw_kaggle.ipynb` — версия с путями Kaggle.

## Установка

Python 3.11 или 3.12, NVIDIA GPU.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux:

```bash
source .venv/bin/activate
```

Установить `torch` и `torchvision` с CUDA командой из [селектора PyTorch](https://pytorch.org/get-started/locally/). Нужен PyTorch 2.3 или новее.

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Последняя команда должна вывести `True`.

## Данные

Распаковать `quick_draw_input.zip` в `input/` рядом с ноутбуком:

```text
input/
  baseline.pth
  classes.npy
  train_images.npy
  train_labels.npy
  val_images.npy
  val_labels.npy
  test_images.npy
```

Архив передаётся отдельно. Данные, веса и результаты исключены из Git.

## Запуск

Из этой папки:

```bash
jupyter lab quick_draw.ipynb
```

Запустить все ячейки. Или запустить скрипт:

```bash
python train.py
```

В начале кода задаются `batch_size`, `compare_epochs` и `finetune_epochs`. При нехватке видеопамяти уменьшить `batch_size` с 512 до 256 или 128.

В локальном ноутбуке `num_workers=0`, чтобы он работал и на Windows.

Оба ResNet обучаются с нуля на одном train и проверяются на одном validation. Каждый получает 20 эпох, победитель — ещё 30. В submission используется модель с наибольшей validation accuracy, включая baseline. Обучение использует только выданный подсэмпл соревнования.

Результаты сохраняются в `output/`: веса, `history.csv`, `comparison.csv`, `scores.csv` и `submission.csv`.

Baseline проверен на полном validation: accuracy 0.67805. Новые модели ещё не обучались на GPU.
