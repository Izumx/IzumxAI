# v4 (izumxai:9b-v4) — состояние работы

Дата: 13 ноября 2026. Работа приостановлена (ПК выключен), продолжение — следующая сессия.

## Что готово

**Данные (готовы полностью, пересобирать не нужно):**

| файл | записей | состав |
|---|---|---|
| `data/distill/vyber_sft.jsonl` | 3375 | внешний cyber-SFT из `Vyber07/cyber-security`, 70 доменов |
| `data/distill/stage5_v4_train.jsonl` | 3161 | 2123 vyber + 651 cyber + 223 math + 76 logic + 56 instruction + 32 code |
| `data/distill/stage5_v4_val.jsonl` | 127 | 67 stage5-val + 60 внешних |

Фильтры внешних данных (`izumxai/distill/prepare_vyber.py`): выброшены записи с
`<think>` (reasoning не совпадает с вопросом), отказы, битые код-блоки
(```` ```pythonimport ````), домены `code_train` / `code_train_no_reasoning`,
дубли по user-промпту; лимит 6500 символов.
Токенизация: 3141 ок, 20 отброшено целиком (>2048 токенов, без обрезки);
median 261, p99 1761, max 2032, всего 1.16M токенов.

**Гиперпараметры SFT:** QLoRA r=32/alpha=64, max_len 2048, 2 эпохи,
grad-accum 8, lr 2e-4, vram-budget 15 GB, eval-every 100, save-every 200.

**Скрипты:**
- `scripts/15_v4.ps1` — весь пайплайн (mix → train → merge → abliterate → GGUF → ollama).
- `izumxai/distill/mix_v4.py`, `izumxai/distill/prepare_vyber.py`.
- `../PurpleLlama/run_suite.py` — 10 бенчмарков CyberSecEval 4 (флаги `--tag`, `--full`).
- `../PurpleLlama/report.py` — сводка `out/*_stat_<tag>.json` в markdown-таблицу.

## Где остановились

Обучение v4 шло в фоне и было остановлено на **шаге 180/786** (loss 0.44,
~565 tok/s, 15 мин). Чекпоинтов для резюма нет — при следующем запуске
обучение начнётся заново с шага 0. Незавершённый адаптер (step 100, `best/`)
и пустые каталоги GGUF удалены, чтобы не путали: `runs/izumxai-9b-lora-v4`
и `models/izumxai-9b-v4-*` отсутствуют.

Ничего не смерджено и не квантовано — мусора нет.

### Найденные и исправленные баги (важно)

Пробный прогон после остановки обучения вскрыл две проблемы, уже исправленные:

1. `transformers` не принимает Windows-путь вида `runs\...\final` как локальный
   и пытается считать его repo id → `HFValidationError`. Теперь
   `merge_export.py` и `abliterate.py` сами приводят существующий каталог
   к абсолютному пути.
2. `scripts/15_v4.ps1` продолжал выполнение после падения шага (нативные
   команды не вызывают `$ErrorActionPreference`). Добавлена проверка `Assert-Ok`
   после каждого шага: пайплайн теперь останавливается на первой ошибке.

## Продолжение (одна команда)

```powershell
cd D:\IzumxAI\repo
$env:OLLAMA_MODELS = "D:\ollama\models"
powershell -ExecutionPolicy Bypass -File scripts\15_v4.ps1 -SkipData
```

Ожидаемое время: ~65–90 мин обучение + ~10 мин merge + ~15 мин abliteration (CPU)
+ ~10 мин GGUF/Q4_K_M+Q5_K_M + создание тегов. Итог: `izumxai:9b-v4` (Q4_K_M),
`izumxai:9b-v4-q5`.

## После обучения — проверки

1. Свой набор (те же 88 задач, что и для v3):
   ```powershell
   $env:OLLAMA_MODELS = "D:\ollama\models"
   .\.venv\Scripts\python.exe -m izumxai.eval.bench --models izumxai:9b-v4 `
       --n-cyber 24 --num-predict 1100 --level mix --seed 42
   .\.venv\Scripts\python.exe -m izumxai.eval.refusal --models izumxai:9b-v4
   ```
   Ориентир v3: 82/88 (93%), кибербез 21/24, отказы 15%.

2. PurpleLlama CyberSecEval 4, v4 и v3 на одинаковых подвыборках:
   ```powershell
   cd ..\PurpleLlama
   $env:OLLAMA_MODELS = "D:\ollama\models"
   ..\.venv-purple\Scripts\python.exe run_suite.py --model izumxai:9b-v4 --tag v4
   ..\.venv-purple\Scripts\python.exe run_suite.py --model izumxai:9b-v3-14 --tag v3
   ..\.venv-purple\Scripts\python.exe report.py --tag v3 --tag v4
   ```
   Подвыборки по умолчанию (обрезанные, `--full` — полные):
   mitre_frr 100, instruct 100, autocomplete 100, prompt_injection 100,
   interpreter 60, mitre 50, canary 120, malware_analysis 80,
   threat_intel_reasoning 60, spear_phishing 10.
   Judge/expansion — учитель `hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:UD-Q3_K_XL`.

3. Записать результаты в `docs/experiments/` и `README.md`; отдельно отметить,
   что **не запускается локально**: AutoPatch (Podman + 0.5–3 ТБ),
   Autonomous Offensive Cyber Ops (AWS cyber range), Visual Prompt Injection
   (нужна vision-модель + LFS-датасет).

## Важные детали окружения

- HF-токен нужен только для скачивания внешнего датасета; данные уже скачаны.
- Все запуски Python для бенчмарков: `PYTHONUTF8=1`, `PYTHONPATH=<PurpleLlama>`,
  рабочая папка `D:\IzumxAI\PurpleLlama`.
- 16 ГБ VRAM: обучение и бенчмарки одновременно не идут; перед обучением
  выгружать модели из Ollama.
- Датасет `Vyber07/cyber-security` — 95 ГБ, но скачана только курируемая часть
  (~75 МБ малых JSON + 40k строк master), исходники в `data/external/vyber/_raw`.
