FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ENV DEBIAN_FRONTEND=noninteractive
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

ADD docker/entrypoint.sh /tmp
ENTRYPOINT ["bash", "/tmp/entrypoint.sh"]
