# Quick, Draw!

`train.py` проверяет выданный baseline на полном validation, сравнивает два ResNet18, дообучает лучший и сохраняет submission. `quick_draw.ipynb` запускает тот же код из Jupyter.

Оба ResNet начинают с `baseline.pth`. Входная свёртка имеет размер 3×3 или 7×7, stride=1; начальный max pooling убран. Для 3×3 используется центральная часть исходного ядра. Остальные веса сохраняются. Изображения остаются 28×28.

`quick_draw_kaggle.ipynb` — прежний эксперимент с моделями ширины 32 и 64, обучаемыми с нуля.

## Окружение

На домашнем ПК используется Python 3.14, PyTorch 2.13.0, torchvision 0.28.0, CUDA 13.0 и RTX 4070 Ti. PyTorch устанавливается из [официального CUDA-репозитория](https://pytorch.org/get-started/previous-versions/).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Если в Ubuntu отсутствует `ensurepip`, создать окружение с `--without-pip`, скачать `https://bootstrap.pypa.io/get-pip.py` и выполнить его через `.venv/bin/python` перед установкой зависимостей.

## Данные

Распаковать `quick_draw_input.zip` в `input/`:

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

## Запуск по SSH

```bash
ssh ubuntu-tiger
cd /home/timk/computer-vision-2026
mkdir -p output/run
nohup .venv/bin/python -u train.py --minutes 165 --output output/run > output/run/train.log 2>&1 < /dev/null &
```

Обучение продолжится после отключения SSH. Ход работы:

```bash
tail -f output/run/train.log
```

Для нового запуска указать новую папку `--output`. При нехватке видеопамяти добавить `--batch-size 256` или `128`.

Проверка всего процесса на небольшой выборке:

```bash
.venv/bin/python train.py --smoke --minutes 2 --output output/smoke
```

Из Jupyter:

```bash
.venv/bin/jupyter lab quick_draw.ipynb
```

## Эксперименты

Обучение использует только train. Каждый вариант получает одинаковый лимит времени и не более 40 эпох: AdamW, lr=0.0003, weight decay=0.0001, cosine schedule, label smoothing=0.05. Победитель получает до 30 дополнительных эпох с lr=0.0001. Часть времени оставляется на проверку и inference.

Поворот до 10°, сдвиг до 5%, масштаб 0.9–1.1 и горизонтальное отражение применяются отдельно к каждому изображению на GPU. Используется AMP. На каждой эпохе вычисляется accuracy на всём validation без аугментаций, сохраняются метрики и лучшие веса.

В конце сравниваются одиночные модели, усреднение original/flip и ансамбль двух вариантов. Способ получения submission выбирается по validation, test используется только для предсказаний.

В папке запуска сохраняются `history.csv`, `scores.csv`, веса, `config.json`, копия исходного кода, `summary.json`, вероятности на validation и `submission.csv`.

В условии порог для 6 баллов — строго выше 0.76102, для 8 — строго выше 0.77489. Отдельного порога для 7 нет. Проверка этих порогов на validation помогает выбрать модель; баллы зависят от результата на Kaggle.

Исходный baseline на полном validation на GPU: 70182/103500 = 0.67808696. CPU-проверка дала 70178/103500 = 0.67804831.

## Дополнительная регуляризация

Если первого эксперимента недостаточно, `improve.py` обучает ResNet18 на входе 56×56 с исходными stride и pooling, начиная с `baseline.pth`. Изображения увеличиваются билинейно. Используются Mixup с alpha=0.2 и AdamW с weight decay=0.01.

```bash
.venv/bin/python -u improve.py --parent output/run --output output/resize56 --minutes 60
```

Второй вариант дообучает лучший checkpoint первого эксперимента с той же регуляризацией:

```bash
.venv/bin/python -u improve.py --parent output/run --output output/regularized --kind parent --minutes 45
```

В конце BatchNorm пересчитывается на 51200 случайных train-примеров без аугментаций. Изменение сохраняется только при улучшении validation. Затем проверяются flip TTA и ансамбль с лучшим результатом первого эксперимента.

Mixup смешивает две картинки и вычисляет loss как взвешенную сумму потерь по их меткам. В логах train accuracy считается по метке картинки с большим весом. Основной критерий сравнения — полный validation.

## Финальный ансамбль

`ensemble.py` сравнивает сохранённые модели и пары моделей на полном validation. Для каждой пары проверяются веса 0.25/0.5/0.75 и температуры 1/2. Температура меняет уверенность предсказаний, не обучая веса сети. Выбранная комбинация целиком сохраняется в `summary.json` и применяется к test.

```bash
.venv/bin/python ensemble.py --runs output/run-20261014 output/resize56-20261014 output/regularized-20261014 --output output/ensemble-20261014
```

В `--runs` перечислить только завершённые запуски с `summary.json`. В результатах указаны точное число верных ответов и размер validation.

## Проверка и просмотр результатов

```bash
.venv/bin/python verify.py output/run-20261014
.venv/bin/jupyter lab results.ipynb
```

`verify.py` заново вычисляет полный validation и все test-предсказания из сохранённых весов. Проверяются совпадение метрики, порядок id и полное совпадение submission. `results.ipynb` показывает только завершённые основные эксперименты; проверочные маленькие выборки в него не входят.

## Архив для передачи

После независимой проверки лучшего запуска:

```bash
.venv/bin/python verify.py output/ensemble-20261014
.venv/bin/python pack.py output/ensemble-20261014
```

В архив входят код, ноутбуки, выданный baseline, лучшие checkpoints каждого завершённого эксперимента и реальные логи/метрики. Датасет и служебные материалы не включаются. Размеры и SHA256 каждого файла записаны в манифест; целостность ZIP проверяется при сборке.

Для Mixup сохраняются последние веса и snapshots каждой десятой эпохи. Они сравниваются на полном validation до и после пересчёта BatchNorm на train. Это позволяет выбрать полезные новые веса, даже когда статистика BatchNorm на смешанных картинках ухудшала промежуточную оценку. Выбор записывается в `calibration.csv`.
