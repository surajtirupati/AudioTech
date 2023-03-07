FROM python:3.7-slim
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install -r requirements.txt --no-cache-dir
RUN apt-get update && apt-get install ffmpeg -y
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 app.main:app