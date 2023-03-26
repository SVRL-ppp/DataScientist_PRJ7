FROM --platform=linux/amd64 node:14.17.0-alpine@sha256:f07ead757c93bc5e9e79978075217851d45a5d8e5c48eaf823e7f12d9bbc1d3c
FROM python:3.8

RUN mkdir /backend 

RUN chmod -R 777 /backend

COPY requirements-backend.txt /backend/requirements-backend.txt

WORKDIR /backend

RUN pip install -r requirements-backend.txt

COPY . /backend

ARG PORT=5000

CMD uvicorn fast_api:app --host 0.0.0.0 --port $PORT --reload