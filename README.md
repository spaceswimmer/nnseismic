# nnseismic

**nnseismic** — репозиторий с кодом для глубинного обучения на сейсмических данных. Здесь реализованы нейросетевые модели для предсказания Relative Geological Time (RGT) по сейсмическим кубам. Визуализации некоторых результатов доступны в [`data/images`](data/images) (интерактивные 3D-вьюверы HTML и статичные изображения).

## Структура репозитория

- **[`src/RGTnet`](src/RGTnet)** — полная реализация нейронной сети из статьи [«3-D Seismic RGT...»](https://doi.org/10.1109/TGRS.2021.3126028). Включает 3D-архитектуру, пайплайн обучения и инференса.
- **[`src/DNN`](src/DNN)** — облегчённая версия сети RGTnet, разработанная мной. Быстрее и проще, при этом сохраняет приемлемое качество предсказаний.
- **[`synthoseis/`](synthoseis)** — сабмодуль, форк [sede-open/synthoseis](https://github.com/sede-open/synthoseis), расположенный по адресу [spaceswimmer/synthoseis](https://github.com/spaceswimmer/synthoseis). Используется для генерации синтетических сейсмических данных.
- **[`data/`](data)** — сейсмические кубы, синтетические данные, сохранённые модели и изображения.
- **[`config/`](config)** — конфигурационные файлы для запуска моделирования.

## Клонирование

```bash
git clone --recurse-submodules git@github.com:spaceswimmer/nnseismic.git
```

Если репозиторий уже склонирован без сабмодулей:

```bash
git submodule update --init --recursive
```

## Запуск

Проект использует [`uv`](https://docs.astral.sh/uv/) для управления зависимостями (см. [`pyproject.toml`](pyproject.toml) и [`uv.lock`](uv.lock)). Требуется Python ≥ 3.9.

```bash
# Установка зависимостей
uv sync

# Запуск Jupyter-ноутбуков
uv run jupyter notebook src/

# Запуск python-файлов
uv run python file.py
```

## Полезные ссылки

- Основной репозиторий: [github.com/spaceswimmer/nnseismic](https://github.com/spaceswimmer/nnseismic)
- Форк synthoseis: [github.com/spaceswimmer/synthoseis](https://github.com/spaceswimmer/synthoseis)
- Оригинальный synthoseis: [github.com/sede-open/synthoseis](https://github.com/sede-open/synthoseis)
