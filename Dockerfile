
FROM ubuntu:latest
RUN mkdir -p /usr/share/tacle
COPY . /usr/share/tacle
RUN apt-get -y update && \
    apt install python3.10 && \
    apt install python3-pip && \
    python3.8 -m pip install pip --upgrade
WORKDIR usr/share/tacle
RUN pip install -r requirements.txt
CMD ["pipeline.sh", "sh"]
