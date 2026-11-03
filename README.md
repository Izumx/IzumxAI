# IzumxAI

Своя языковая модель: `IzumxAI-nano` (собственный transformer, обучение с нуля)
и `izumxai:2b` (компактная модель, которая на узких задачах обгоняет qwen3:8b).

## Честно о физике

Обучить 2–3B «с нуля и лучше qwen3:8b» на одной RTX 4050 (6 ГБ VRAM) нельзя:
фронтирные 8B-модели — это триллионы токенов и тысячи GPU-часов, а не 6 ГБ.
Поэтому здесь два честных пути, оба рабочие на твоём железе:

| Путь | Что это | Параметры | Реальность |
|---|---|---|---|
| `IzumxAI-nano` | своя архитектура + свой BPE, pretrain с нуля | 33M / 104M / 250M | обучается за часы, качество «малыш», но всё своё |
| `izumxai:2b` | Qwen3-1.7B + QLoRA-дистилляция от qwen3:8b | 1.7B | влезает в 6 ГБ, обгоняет 8B на том, на чём учили |

Ключевая идея второго пути: **учитель не является источником истины**.
Каждый ответ qwen3:8b проверяется программно (математика — точное сравнение,
код — реальный запуск тестов). В обучение попадают только верифицированные
пары. Так 2B-модель наследует не галлюцинации учителя, а его рабочие решения.

## Структура

```
izumxai/
  tokenizer.py        свой байтовый BPE (обучение, encode/decode)
  model.py            своя decoder-only модель: RMSNorm, RoPE, GQA, SwiGLU
  train_tokenizer.py  обучение токенизатора
  build_bin.py        текст -> бинарный поток токенов
  dataset.py          memmap-датасет для pretrain
  train_pretrain.py   цикл pretrain (AdamW, cosine, bf16, чекпоинты)
  chat.py             чат со своим чекпоинтом
  fetch_corpus.py     маленький корпус (tiny-shakespeare + ru-wiki)
  distill/
    gen_data.py       генерация данных от учителя + программная верификация
    train_lora.py     QLoRA-дистилляция в Qwen3-1.7B (6 ГБ VRAM)
    merge_export.py   merge LoRA -> safetensors -> GGUF -> Modelfile
    common.py         шаблон чата, маскирование лосса, VRAM-метрики
  eval/bench.py       IzumxAI:2B vs qwen3:8b: математика, код, скорость
scripts/              01_setup ... 05_bench (PowerShell)
```

## Быстрый старт

```powershell
# 1. окружение (torch cu128 + transformers/peft/bitsandbytes)
powershell -ExecutionPolicy Bypass -File scripts\01_setup.ps1

# 2. своя модель с нуля (~1-2 часа на 4050)
powershell -ExecutionPolicy Bypass -File scripts\02_nano.ps1

# 3. IzumxAI:2B: данные от учителя + QLoRA
powershell -ExecutionPolicy Bypass -File scripts\03_distill.ps1 -N 400

# 4. экспорт в Ollama как izumxai:2b
powershell -ExecutionPolicy Bypass -File scripts\04_export.ps1

# 5. честный замер против qwen3:8b
powershell -ExecutionPolicy Bypass -File scripts\05_bench.ps1
```

## Пайплайн nano

```powershell
.\.venv\Scripts\python.exe -m izumxai.train_tokenizer --raw data/raw --out data/tokenizer --vocab 8192
.\.venv\Scripts\python.exe -m izumxai.build_bin --raw data/raw --tokenizer data/tokenizer/tokenizer.json --out data/bin/corpus.bin
.\.venv\Scripts\python.exe -m izumxai.train_pretrain --data data/bin/corpus.bin `
    --tokenizer data/tokenizer/tokenizer.json --preset nano-30m --steps 5000 --out runs/nano30m
```

Пресеты модели (см. `izumxai/model.py`): `nano-30m`, `nano-100m`, `nano-250m`.
Параметры и расход памяти считаются через `memory_report(Config.preset(...))`.

## Пайплайн 2B (дистилляция)

1. `gen_data.py` — учитель qwen3:8b решает задачи трёх типов:
   * математика/логика — ответ сверяется с эталоном по числу;
   * код — извлечённый код реально исполняется на тестах в subprocess;
   * русские инструкции — проверяется формат ответа.
2. `train_lora.py` — 4-bit база Qwen3-1.7B, LoRA r=32 на всех линейных слоях,
   gradient checkpointing, batch=1 × grad_accum=16. Пик VRAM ≈ 4.5–5.5 ГБ.
3. `merge_export.py` — слияние адаптера, `convert_hf_to_gguf.py`, `Modelfile`.

Ожидаемый результат (то, что надо подтвердить бенчмарком): на задачах своего
распределения 1.7B с дистилляцией обгоняет 8B, потому что 8B не решает их
пошагово так, как её научили; на широких знаниях 8B всё ещё сильнее.

## Публикация модели

```powershell
ollama create izumxai:2b -f models\izumxai-2b\Modelfile
ollama run izumxai:2b
```

## Ограничения

* Дистилляция сужает модель: она хороша на том, что было в данных, и слабеет на открытых темах.
* qwen3:8b как учитель при 6 ГБ VRAM работает медленно (~2–6 tok/s) — 400 примеров это несколько часов.
* nano-модель не конкурент 8B — это демонстрация, что весь стек (токенизатор, архитектура, обучение) наш.

## Дальше

* больше типов задач в `gen_data.py` (агентные вызовы инструментов, SQL, тесты);
* DPO/ORPO на парах «плохой/хороший ответ»;
* дистилляция reasoning-трейсов (`--think` у учителя) и сжатие их в короткий ответ.
