FROM ubuntu:24.04

RUN apt update && apt -y upgrade && \
    apt install -y python3-pip python3-venv && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    python3 -m pip install -U pip && \
    pip3 install -U setuptools wheel

RUN . venv/bin/activate && pip3 install ai-edge-litert ultralytics torch torchvision opencv-python tensorflow>=2.0.0
RUN /usr/bin/bash -c "echo . ./venv/bin/activate >> /root/.bashrc"
RUN mkdir -p /work/share
RUN apt install -y libgl1 libglib2.0-0

WORKDIR /work
COPY conv.py /work/conv.py
COPY conv.sh /work/conv.sh
RUN chmod +x /work/conv.sh

CMD [ "/usr/bin/bash", "-c", "/work/conv.sh" ]
