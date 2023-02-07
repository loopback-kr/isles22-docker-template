FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN pip install -U pip
COPY requirements.txt /opt/algorithm/
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=algorithm:algorithm models /opt/algorithm/models
COPY --chown=algorithm:algorithm sources /opt/algorithm/sources
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
