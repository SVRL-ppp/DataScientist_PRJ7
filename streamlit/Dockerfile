FROM --platform=linux/amd64 node:14.17.0-alpine@sha256:f07ead757c93bc5e9e79978075217851d45a5d8e5c48eaf823e7f12d9bbc1d3c

FROM python:3.8.15

CMD mkdir /stapp
COPY . /stapp

WORKDIR /stapp

EXPOSE 8501

RUN pip install -r requirements.txt

CMD streamlit run dashboard.py --server.port $PORT