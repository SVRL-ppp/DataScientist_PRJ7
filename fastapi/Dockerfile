FROM --platform=linux/amd64 node:14.17.0-alpine

FROM python:3.8.15

CMD mkdir /fapi

COPY . /fapi

WORKDIR /fapi

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
