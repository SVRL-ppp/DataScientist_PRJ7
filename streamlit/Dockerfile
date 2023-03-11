FROM --platform=linux/amd64 node:14.17.0-alpine

FROM python:3.8.15

CMD mkdir /stapp
COPY . /stapp

WORKDIR /stapp

EXPOSE 8501

RUN pip install -r requirements.txt

CMD streamlit run dashboard.py --server.port $PORT