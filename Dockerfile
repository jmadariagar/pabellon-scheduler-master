FROM python:3.6
ENV PYTHONUNBUFFERED 1

# Installing OS Dependencies
#RUN apt-get update && apt-get upgrade -y && apt-get install -y libsqlite3-dev

RUN pip install -U pip setuptools

RUN mkdir /config
COPY pabellonScheduler /config
RUN pip install -r /config/requirements.txt

RUN mkdir /src
WORKDIR /src
