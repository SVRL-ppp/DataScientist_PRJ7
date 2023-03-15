FROM --platform=linux/amd64 node:14.17.0-alpine@sha256:f07ead757c93bc5e9e79978075217851d45a5d8e5c48eaf823e7f12d9bbc1d3c

FROM python:3.8.15@sha256:7c5b3ec91b7e4131dead176d824ddf317e6c19fe1ae38c54ec0ae6038908dd18

CMD mkdir /stapp
COPY . /stapp

WORKDIR /stapp

EXPOSE 8501

RUN pip install -r requirements.txt

CMD streamlit run dashboard.py --server.port $PORT