FROM nvidia/cvdp-sim:v1.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_UID=1006
ARG HOST_GID=117

RUN groupadd -g ${HOST_GID} mage && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash mage && \
    mkdir -p /app /code /tmp/mage && \
    chown -R mage:mage /app /code /tmp/mage

COPY agent.py /app/agent.py
RUN chmod 755 /app/agent.py && chown mage:mage /app/agent.py

USER mage
ENV HOME=/home/mage
WORKDIR /code
ENTRYPOINT ["python3", "/app/agent.py"]
