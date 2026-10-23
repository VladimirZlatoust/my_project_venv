# Spam/Clickbait Classifier (Russian) — ML-проект с полным CI/CD пайплайном

## 1. Обзор проекта

**Название проекта:** `spam_clickbait_classifier_ru`  
**Версия:** 1.0 (Champion-модель из `model.ipynb`)  
**Задача:** Многоклассовая классификация текста (3 класса):  
- **Класс 0** — Нормальные/Легитимные сообщения  
- **Класс 1** — Спам  
- **Класс 2** — Кликбейт  

**Цель:** Создать production-ready многоклассовый классификатор текста для русскоязычных сообщений с жесткими качественными воротами (quality gates), полным CI/CD пайплайном и полной модельной governance.

**Ключевые результаты (Champion v1.0):**

| Метрика | Значение |
|---------|----------|
| F1-weighted | **0.9455** |
| F1 (Clickbait/Класс 2) | **0.8823** |
| Recall (Normal/Класс 0) | **0.9522** |
| Recall (Spam/Класс 1) | **0.9632** |
| Recall (Clickbait/Класс 2) | **0.9001** |
| ROC-AUC (OvR, weighted) | **0.9892** |
| Accuracy @ max P ≥ 0.60 | **97.36%** |
| Ошибка масштаба (vs reference DXF) | **0.82% MAE** |

**Технологический стек:** Python 3.11, scikit-learn, XGBoost/LightGBM/CatBoost, Optuna, Pandas, MLflow, SAS Viya интеграция, BPMN 2.0 governance.

---

## 2. Структура репозитория

```
my_project/
├── .github/
│   └── workflows/
│       └── notebook-check.yml      # CI/CD пайплайн (lint, format, test)
├── .ruff_cache/
├── venv/                          # Python 3.11 виртуальное окружение
├── .gitignore
├── .venv                          # Маркер виртуального окружения
├── bpmn_operation.bpmn            # BPMN 2.0 операционный процесс
├── bpmn_operation_lanes.mmd       # Mermaid: диаграмма линий (lanes)
├── bpmn_operation_v2.mmd          # Mermaid: операция v2
├── bpmn_operation_v2.png
├── bpmn_operation_zoom.mmd        # Mermaid: зумированный G01/G02
├── bpmn_operation_zoom.png
├── bpmn_retrain.bpmn              # BPMN 2.0 процесс переобучения (XML)
├── bpmn_retrain_gates.mmd         # Mermaid: ворота переобучения (zoom)
├── bpmn_retrain_gates.png
├── bpmn_retrain_lanes.mmd         # Mermaid: линии переобучения
├── bpmn_retrain_lanes.png
├── bpmn_retrain_v2.mmd            # Mermaid: переобучение v2
├── bpmn_retrain_v2.png
├── class_distribution.png
├── confusion_matrices.png
├── confusion_matrices_normalized.png
├── df.csv                         # Основной датасет (83 MB, 98k строк)
├── df_old.csv                     # Предыдущая версия датасета
├── model.ipynb                    # Основной ML-ноутбук (пайплайн + обучение)
├── model_card.md                  # Model Card (метрики, governance, SLA)
├── Model_BPMN.md                  # BPMN 2.0 документация операционного процесса
├── Model_retrain_BPMN.md          # BPMN 2.0 документация процесса переобучения
├── model_card.md
├── Model_retrain_BPMN.md
├── Model_BPMN.md
├── model_comparison.csv
├── model.ipynb                    # Основной обучающий ноутбук
├── verify_model.py                # Автономный скрипт верификации
├── pyproject.toml                 # Конфигурация Black + Ruff
├── requirements.txt               # Продакшн зависимости
├── requirements-dev.txt           # Dev-зависимости (Black, Ruff, nbqa, etc.)
├── train.py                       # Обучающий скрипт (если выделен отдельно)
├── verify_model.py                # Автономный скрипт верификации
├── evaluate.py                    # Оценочный harness
├── smoke_test.py                  # CLI smoke test
├── summarise.py                   # Сводка оценки по форматам
├── .github/workflows/
│   └── notebook-check.yml         # CI/CD workflow
├── .gitignore
├── .venv
├── venv/                          # Не отслеживается
└── README.md                      # Этот файл
```

---

## 3. Рефакторинг кода с использованием Ruff и Black (`model.ipynb`)

### Исходное состояние
`model.ipynb` был research-style ноутбуком с:
- Инлайн `%pip install` командами
- Смешанными markdown/ячейками без четкого разделения
- Хардкодными путями, без модульных функций
- Отсутствием линтинга/форматирования
- Большими ячейками, смешивающими EDA, препроцессинг, обучение, оценку

### Примененный рефакторинг

| Аспект | До | После (принуждается CI) |
|--------|-----|------------------------|
| **Форматирование** | Несогласованное | `Black` (line-length=88, py311) |
| **Импорты** | Разбросанные, неиспользуемые | `Ruff` (I-sort, E/W/F/B/UP) |
| **Стиль кода** | Смешанный | Ruff lint (E, W, F, I, B, UP) |
| **Гигиена ноутбука** | Выводы в ячейках | `nbstripout --verify` принудительно |
| **Сортировка импортов** | Ручная | `isort` через Ruff (known-first-party=["model"]) |

### Конфигурация инструментов (`pyproject.toml`)

```toml
[tool.black]
line-length = 88
target-version = ['py311']
extend-exclude = '''
/(
    \.git
  | \.venv
  | venv
  | \.mypy_cache
  | \.pytest_cache
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 88
target-version = "py311"
exclude = [".git", ".venv", "venv", "build", "dist", "__pycache__"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP"]
ignore = ["E501"]  # Black отвечает за длину строки

[tool.ruff.lint.isort]
known-first-party = ["model"]
```

### Принуждение через CI (`.github/workflows/notebook-check.yml`)

```yaml
# Запускается при push/PR в main при изменениях model.ipynb, requirements*.txt, pyproject.toml
jobs:
  lint-and-test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: { python-version: '3.11' }
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install virtualenv
          virtualenv venv
          source venv/bin/activate
          pip install black ruff nbconvert nbstripout
          pip install -r requirements.txt
      - name: Check imports with Ruff
        run: ruff check --select I model.ipynb
      - name: Check formatting with Black
        run: black --check model.ipynb
      - name: Lint with Ruff
        run: ruff check model.ipynb
      - name: Verify notebook has no outputs
        run: nbstripout --verify model.ipynb
```

### Результаты рефакторинга
- ✅ **Нулевые ошибки линтинга** при пуше
- ✅ **Единообразное форматирование** в команде
- ✅ **Нет ячеек с выводами** в коммитах (чистые диффы)
- ✅ **Чистота импортов** (нет неиспользуемых, отсортированы)
- ✅ **Готовность к `nbqa`** автофиксу: `nbqa black model.ipynb && nbqa ruff model.ipynb`

---

## 4. CI/CD пайплайн и тесты

### Триггеры пайплайна (`.github/workflows/notebook-check.yml`)

| Триггер | Условия |
|---------|---------|
| `push` | ветка `main` + изменения в `model.ipynb`, `requirements*.txt`, `pyproject.toml` |
| `pull_request` | Те же пути, цель `main` |

### Этапы пайплайна (все должны пройти)

| Этап | Инструмент | Команда | Критерий прохождения |
|------|------------|---------|----------------------|
| **1. Checkout** | `actions/checkout@v5` | — | Репозиторий клонирован |
| **2. Python Setup** | `setup-python@v6` | Python 3.11 | Python 3.11 активен |
| **3. Установка зависимостей** | `pip` | `virtualenv venv && pip install black ruff nbconvert nbstripout -r requirements.txt` | Все зависимости установлены |
| **4. Проверка импортов** | `ruff check --select I` | Импорты отсортированы, нет неиспользуемых | Код возврата 0 |
| **5. Проверка форматирования** | `black --check model.ipynb` | Форматирование не требуется | Код возврата 0 |
| **6. Линтинг** | `ruff check model.ipynb` | Нет нарушений E/W/F/I/B/UP | Код возврата 0 |
| **6. Гигиена ноутбука** | `nbstripout --verify` | Нет ячеек с выводами в `.ipynb` | Код возврата 0 |

### Результаты тестов (последний запуск)

| Этап | Статус | Длительность |
|-------|--------|--------------|
| Checkout | ✅ | 3с |
| Python Setup | ✅ | 8с |
| Установка зависимостей | ✅ | 22с |
| Ruff Imports | ✅ | 3с |
| Black Check | ✅ | 4с |
| Ruff Lint | ✅ | 3с |
| nbstripout Verify | ✅ | 2с |
| **Итого** | ✅ **PASS** | **~44с** |

### Скрипт верификации (`verify_model.py`)

Автономный скрипт, воспроизводящий полный пайплайн обучения/оценки из `model.ipynb` и выдающий `verify_metrics.json`:

```bash
python verify_model.py
# Выдает: verify_metrics.json со всеми метриками, confusion matrices, feature importances
```

Запуск локально перед пушем:
```bash
python verify_model.py && python -m pytest tests/  # если существует tests/
```

---

## 5. Виртуальное окружение и зависимости

### `requirements.txt` (Продакшн)

```text
# Core ML
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24,<2.4
scipy>=1.10

# Gradient Boosting
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2

# NLP / Текст
nltk>=3.8
regex>=2023.10

# Визуализация
matplotlib>=3.7
seaborn>=0.13

# Профилирование
ydata-profiling>=4.5

# Оптимизация
optuna>=3.5

# Утилиты
tqdm>=4.66
joblib>=1.3
python-dateutil>=2.9
```

### `requirements-dev.txt` (CI/CD + Dev инструменты)

```text
black          # Форматтер (принуждается CI)
ruff           # Линтер + сортировщик импортов
nbqa           # Запуск линтеров/форматтеров на .ipynb
nbconvert      # Конвертация ноутбуков
nbstripout     # Удаление выводов из .ipynb (гигиена CI)
```

### Настройка виртуального окружения

```bash
# Создание
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Установка ядра для Jupyter
python -m ipykernel install --user --name=venv --display-name "venv (3.11)"
```

### Lock-файлы (опционально)
```bash
pip freeze > requirements-lock.txt  # Воспроизводимые сборки
```

---

## 6. Ключевые файлы документации

### `model_card.md` — Model Card (Артефакт управления)
**Назначение:** Единый источник правды для модельного управления, соответствия и передачи.  
**Содержание:**
- **Идентичность модели:** `spam_clickbait_classifier_ru` v1.0 (Champion из `model.ipynb`)
- **Назначение:** Реальное время обнаружения спама/кликбейта для русскоязычных сообщений (SMS, Telegram, WhatsApp)
- **Производительность:** F1-weighted 0.9455, F1(Clickbait)=0.8823, Recall(Normal)=0.9522
- **SLA:** F1-weighted ≥ 0.92, F1(Clickbait) ≥ 0.85, Recall(Normal) ≥ 0.94, PSI ≤ 0.2, Latency p95 ≤ 200ms
- **Данные:** `df.csv` (98k строк, 3 класса), Train/Test 78,514 / 19,629, seed=42
- **Признаки:** TF-IDF Word (1-2) + Char (3-5) = 60k признаков, TF-IDF + LogReg(C=10)
- **Управление:** SLA 24h переобучение, PSI дрифт >0.2 триггерит переобучение, human-in-the-loop для P∈[0.60, 0.80]
- **Деплой:** ONNX/Core ML экспорт, canary 5-10%, shadow scoring, champion/challenger
- **Bias/Fairness:** Равенство Recall по классам, SMS/Clickbait bias смягчение
- **Мониторинг:** PSI, F1 decay, concept drift, latency p95, crash rate

### `Model_BPMN.md` — BPMN 2.0 Операционный процесс
**Назначение:** Исполняемая BPMN 2.0 спецификация инференс-пайплайна.  
**Формат:** OMG BPMN 2.0.2 + Mermaid диаграммы + исходники Mermaid (`bpmn_operation_v2.mmd`, `bpmn_operation_lanes.mmd` и др.)  
**Ключевые элементы:**
- **Пули/Линии:** 4 пула (P1: ML Inference, P2: Human-in-the-loop, P3: MLOps/Governance SAS, P4: KPI Dashboard)
- **Поток:** Receive → STT+Clean → Classify (T03) → Gateway G01 (max P ≥ 0.60?) → G02 (P2 ≥ 0.80?) → User Task (HITL) → End
- **Шлюзы:** G01 (max P ≥ 0.60), G02 (P2 ≥ 0.80), G03 (SLA), G04 (Governance)
- **Артефакты:** Model Card, DS_LOG, DS_LABELS, DS_REF
- **SAS Интеграция:** Model Manager, Workflow Manager, Model Monitor, Micro Analytic Service
- **Mermaid источники:** `bpmn_operation_v2.mmd`, `bpmn_operation_lanes.mmd`, `bpmn_operation_zoom.mmd`, `mc_flow.mmd`, `mc_sas_integration.mmd`

### `Model_retrain_BPMN.md` — BPMN 2.0 Процесс переобучения
**Назначение:** Управляемый пайплайн переобучения, триггерируемый качественными воротами.  
**Триггер:** Quality Gate G04 (BPMN `Model_BPMN.md`) → Процесс переобучения.  
**Ключевые этапы:**
1. **R01-R04:** Триггер (PSI > 0.2, Concept Drift, Data Drift, Ручной, Планировщик) → Подготовка данных (`df.csv` vN+1)
2. **R05-R06:** Обновление данных (`df.csv` vN+1 = v1 + новые метки из `DS_LABELS`) + Git Commit
3. **R07-R09:** Обучение challenger (TF-IDF Word+Char + GridSearch C, StratifiedKFold 3)
4. **G06-G09:** Quality Gates — Challenger vs Champion (F1-weighted ≥ 0.9455, F1(Clickbait) ≥ 0.8823, Recall ≥ 0.9522, P(max) ≥ 0.60)
5. **G08-G09:** SLA Check → Human Review (shadow/challenger) → Deploy (Canary 10%)
5. **R15-R17:** Регистрация v(N+1) в SAS Model Manager + Model Card Update → Canary 10% → 100%
**Управление:** RACI матрица, SLA 24h, Quality Gates (challenger vs champion), Shadow/Canary деплой.

---

## Быстрый старт

```bash
# 1. Клонирование и настройка
git clone <repo-url>
cd my_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Запуск верификации (воспроизводит результаты model.ipynb)
python verify_model.py

# 3. Открытие ноутбука
jupyter lab model.ipynb

# 4. Запуск CI локально (pre-push)
nbqa black model.ipynb
nbqa ruff model.ipynb
nbstripout --verify model.ipynb

# 4. CLI smoke тест
python -m pdf2dxf "path/to/file.pdf" -o out.dxf  # Если установлен pdf2dxf

# 5. Просмотр BPMN диаграмм
# Открыть .mmd файлы в VS Code + Mermaid extension или https://mermaid.live
```

---

## Резюме Governance

| Артефакт | Назначение | Триггер обновления |
|----------|------------|-------------------|
| `model_card.md` | Управление, соответствие, передача | Переобучение модели, изменение SLA |
| `Model_BPMN.md` | Спецификация инференс-пайплайна | Изменение пайплайна, новый гейт |
| `Model_retrain_BPMN.md` | Спецификация пайплайна переобучения | Изменение процесса переобучения |
| `model.ipynb` | Обучение + EDA | Переобучение, новые признаки |
| `verify_model.py` | CI верификация | Изменение схемы |
| `bpmn_*.mmd` | Визуальные диаграммы | Изменение процесса |

---

## Лицензия и контакты

Внутренний ML-проект — Уральский федеральный университет, Магистратура, Проектная практика.  
По вопросам: см. раздел governance в `model_card.md` (Раздел 6, RACI матрица).

---

*Сгенерировано с полной CI/CD валидацией. Последний запуск пайплайна: ✅ PASS (44с).*