FROM python:3.11-slim

WORKDIR /app

ADD https://storage.yandexcloud.net/hackathon-perm-2026/cte_ids.npy ./models/indexes/cte_ids.npy

ADD https://storage.yandexcloud.net/hackathon-perm-2026/faiss_flat_(metric%3Dip)__intfloat_multilingual-e5-small.index  ./models/indexes/faiss_flat_(metric=ip)__intfloat_multilingual-e5-small.index

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9593

CMD ["uvicorn", "src.api.search_api:app", "--host", "0.0.0.0", "--port", "9593"]

