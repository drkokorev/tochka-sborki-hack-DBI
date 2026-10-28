# Облегчённый образ: веб-интерфейс, фразник, правила и LLM-шлюз. Локальные GigaAM, Silero и ученик —
# в requirements-ml.txt и tools/get_voice.sh (на ВМ хакатона ставятся отдельно, см. README).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p logs data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "apps.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
