FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output &&\
    chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN pip install --no-cache-dir -U pip
COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=algorithm:algorithm models ./models
COPY --chown=algorithm:algorithm sources ./sources
COPY --chown=algorithm:algorithm *.py ./
RUN python remove_orphans.py
RUN rm requirements.txt remove_orphans.py

ENTRYPOINT python -m process $0 $@