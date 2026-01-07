# WiseTask Semantic Search API

Сервис семантического поиска по задачам и плагинам для WiseTask.

API принимает сущности (tasks/plugins), строит для них эмбеддинги с помощью модели `BAAI/bge-m3` из `sentence-transformers`, сохраняет данные в SQLite и позволяет выполнять поиск по смыслу (cosine similarity).

## Возможности

- Индексация задач: загрузка списка задач и сохранение эмбеддингов в SQLite.
- Индексация плагинов: загрузка списка плагинов и сохранение эмбеддингов в SQLite.
- Семантический поиск: поиск наиболее похожих задач/плагинов по текстовому запросу.
- Docker-запуск: быстрый старт без установки окружения локально.

## Запуск (Docker)

Собрать образ и запустить контейнер:

```bash
cd wise-task-search
docker build -t semantic-search-api .
docker run --rm -p 8001:8001 semantic-search-api
```

После старта API будет доступен по адресу:

- http://localhost:8001

Примечание: при первом запуске модель может загружаться/инициализироваться некоторое время — это нормально.

## Проверка работы

### Health check

```bash
curl http://localhost:8001/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

## Быстрые примеры

### Индексация задач (bulk)

```bash
curl -X POST http://localhost:8001/tasks/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      { "id": "1", "name": "Задача про графы", "description": "описание", "category": "Graphs" }
    ]
  }'
```

### Поиск задач

```bash
curl -X POST http://localhost:8001/tasks/search \
  -H "Content-Type: application/json" \
  -d '{ "query": "поиск про графы", "top_k": 3 }'
```

### Индексация плагинов (bulk)

```bash
curl -X POST http://localhost:8001/plugins/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "plugins": [
      {
        "id": "p1",
        "name": "Graph Builder",
        "description": "строит граф",
        "category": "Graphs",
        "graphType": "DAG",
        "pluginType": "builder"
      }
    ]
  }'
```

### Поиск плагинов

```bash
curl -X POST http://localhost:8001/plugins/search \
  -H "Content-Type: application/json" \
  -d '{ "query": "построение графа", "top_k": 3 }'
```
