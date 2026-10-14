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

`ensemble.py` сравнивает сохранённые модели, пары и усреднение трёх или более моделей на полном validation. Для каждой пары проверяются веса 0.25/0.5/0.75 и температуры 1/2/4. Для трёх или более моделей проверяется равномерное усреднение и вес 0.5 у каждой модели по очереди. Температура меняет уверенность предсказаний, не обучая веса сети. Выбранная комбинация целиком сохраняется в `summary.json` и применяется к test.

```bash
.venv/bin/python ensemble.py --runs output/run-20261014 output/resize56-20261014 output/regularized-fixed-20261014 --output output/ensemble-20261014
```

В `--runs` перечислить только завершённые запуски с `summary.json`. В результатах указаны точное число верных ответов и размер validation.

## Проверка и просмотр результатов

```bash
.venv/bin/python verify.py output/ensemble-20261014
.venv/bin/jupyter lab results.ipynb
```

`verify.py` заново вычисляет полный validation и все test-предсказания из сохранённых весов. Проверяются совпадение метрики, порядок id и полное совпадение submission. `results.ipynb` показывает только завершённые основные эксперименты; проверочные маленькие выборки в него не входят.

Для inference используются фиксированные алгоритмы cuDNN, без benchmark и TF32. Если submission был создан со старыми настройками, пересоздать его и отдельно проверить:

```bash
.venv/bin/python verify.py output/ensemble-20261014 --write-submission
.venv/bin/python verify.py output/ensemble-20261014
```

## Архив для передачи

После независимой проверки лучшего запуска:

```bash
.venv/bin/python verify.py output/ensemble-20261014
.venv/bin/python pack.py output/ensemble-20261014
```

В архив входят код, ноутбуки, выданный baseline, лучшие checkpoints каждого завершённого эксперимента и реальные логи/метрики. Датасет и служебные материалы не включаются. Размеры и SHA256 каждого файла записаны в манифест; целостность ZIP проверяется при сборке.

Для Mixup сохраняются последние веса и snapshots каждой десятой эпохи. Они сравниваются на полном validation до и после пересчёта BatchNorm на train. Это позволяет выбрать полезные новые веса, даже когда статистика BatchNorm на смешанных картинках ухудшала промежуточную оценку. Выбор записывается в `calibration.csv`.

## Результат 14 октября

| Вариант | Accuracy на полном val |
| --- | ---: |
| Выданный baseline | 0.67808696 |
| ResNet18, входная свёртка 3×3 | 0.72464734 |
| ResNet18, входная свёртка 7×7 | 0.72141063 |
| Первый ансамбль с flip | 0.74212560 |
| Ансамбль с моделью на входе 56×56 | 0.74389372 |
| Итоговый ансамбль | 0.74911111 |

Итог: 77533 верных ответа из 103500, прирост 7.10 процентного пункта к baseline. Четыре модели усредняются с одинаковыми весами, flip TTA и температурой 4. Дообучение с Mixup не улучшило одиночную модель. Пороги 0.76102 и 0.77489 на val не достигнуты; результата Kaggle пока нет.

Submission содержит 172500 строк. После фиксации алгоритмов inference отдельный процесс повторил полный val и все test-предсказания: метрика совпала, расхождений в submission нет. Реальные результаты и состав ансамбля сохранены в `results.ipynb` и `output/ensemble-20261014/summary.json`.

## ViT с нуля

`vit.py` обучает один ViT на тех же рисунках 28×28: патчи 4×4, 49 патчей, размер признаков 192, 6 блоков, 3 головы attention, усреднение выходных признаков. В модели 2749209 параметров. Начальные веса случайные, внешние данные и pretrained weights не используются.

AdamW с lr=0.001 и weight decay=0.05; bias, LayerNorm и позиционные параметры не получают weight decay. Первые 5 эпох используются для warmup, затем lr убывает по cosine до 0.00001. Число эпох оценивается по времени первой эпохи, обучение ограничено общим временем. Последние десять минут оставлены для проверки и submission.

Аугментации совпадают с ResNet. Дополнительно используются Mixup с alpha=0.2, label smoothing=0.05, DropPath=0.1 и ограничение нормы градиента до 1. Mixup отключается в последних десяти плановых эпохах. EMA с decay=0.999 усредняет параметры в ходе обучения; лучший одиночный checkpoint выбирается между текущими весами и EMA по полному val. Предсказания разных моделей не усредняются.

Для сравнения взят лучший одиночный ResNet из предыдущих прогонов: `output/regularized-fixed-20261014/model.pth`. Перед обучением ViT он заново проверяется на тех же 103500 val-примерах. В итоговой таблице для обеих моделей есть обычный inference и одинаковый flip TTA.

```bash
.venv/bin/python -u vit.py --output output/vit-20261014
.venv/bin/python vit.py --verify --output output/vit-20261014
.venv/bin/jupyter lab vit_results.ipynb
.venv/bin/python pack.py output/vit-20261014 --archive Карсалиев_Тимур_HW_2_ViT.zip
```

Обычный запуск ограничен тремя часами. `--deadline` задаёт абсолютный Unix timestamp окончания. `--smoke` проверяет весь процесс на 512 train и 64 val/test, результаты этой проверки не относятся к соревнованию. Полный запуск сохраняет реальные метрики в отдельной папке; прежние результаты ResNet остаются доступны.

Проверить flip TTA во время обучения можно на отдельной сохранённой копии лучшего checkpoint. Оба варианта используют один полный val:

```bash
.venv/bin/python vit_tta.py --output output/vit-tta-check
```

Копия весов, метрики и номер эпохи сохраняются в указанной новой папке.

Разбор ошибок по классам использует сохранённые вероятности на val:

```bash
.venv/bin/python class_errors.py --predictions output/vit-tta-check/val_flip.npy --output output/class-errors
```

Сохраняются accuracy каждого класса, матрица ошибок, частые путаницы и примеры уверенных ошибочных предсказаний. Разбор относится к конкретному checkpoint, указанному в папке проверки TTA.

## Pretrained ViT

Предыдущий ViT с нуля остановлен по запросу пользователя после 89 эпох. Лучший сохранённый checkpoint: EMA эпохи 87, accuracy 0.74065700 на 103500 val без TTA. Его веса и история сохранены отдельно.

`pretrained_vit.py` дообучает `vit_small_patch16_224.augreg_in21k_ft_in1k`: внешние веса ImageNet-21k → ImageNet-1k из timm. В модели 21601881 параметр, 12 блоков, 6 голов, размер признаков 384. Вход увеличивается с 28×28 до 224×224 бикубически; патчи 16×16, 196 патчей. Рисунок инвертируется в чёрные линии на белом фоне, значения нормализуются через mean=0.5, std=0.5. timm адаптирует входную свёртку к одному каналу, новая голова содержит 345 классов.

Для дообучения используется только выданный train. Внешние pretrained-веса разрешены пользователем отдельно. Val используется для выбора checkpoint и flip TTA, test — для предсказаний.

AdamW: lr=0.0001 для backbone и 0.001 для новой головы, weight decay=0.01, warmup половину эпохи и cosine schedule. Batch size 64, bfloat16 autocast, label smoothing=0.05, EMA=0.99. Из аугментаций используются прежние повороты, сдвиги, масштаб и flip; Mixup отключён. Новый запуск сохраняется отдельно от предыдущих результатов.

```bash
.venv/bin/python -u pretrained_vit.py --output output/pretrained-vit-20261015
.venv/bin/python pretrained_vit.py --verify --output output/pretrained-vit-20261015
.venv/bin/jupyter lab pretrained_results.ipynb
```

`--deadline` задаёт общий лимит абсолютным Unix timestamp. Последние 15 минут оставлены для полного val/test и отдельной проверки. Для inference повторно загружать внешние веса не нужно: модель создаётся с `pretrained=False` и загружает локальный `vit.pth`. Сравнение с лучшим одиночным ResNet использует тот же полный val, отдельно без TTA и с flip TTA.
