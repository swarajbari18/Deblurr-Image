FROM python:3.8.13-slim-buster
COPY . /app
WORKDIR /app
RUN make setup
CMD [ "make", "deployment" ]