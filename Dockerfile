FROM python:3.11.9-slim

WORKDIR /app 

RUN pip install uv
COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY . .

EXPOSE 7000

CMD ["uvicorn", "app.core.api:app", "--host", "0.0.0.0", "--port", "7000"]
