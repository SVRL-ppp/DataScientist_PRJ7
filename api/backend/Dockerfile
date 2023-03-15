FROM --platform=linux/amd64 node:14.17.0-alpine
FROM python:3.8

RUN mkdir /backend 

RUN chmod -R 777 /backend

COPY requirements-backend.txt /backend/requirements-backend.txt

WORKDIR /backend

RUN pip install -r requirements-backend.txt

COPY . /backend

ARG PORT=5000

CMD uvicorn fast_api:app --host 0.0.0.0 --port $PORT --reload