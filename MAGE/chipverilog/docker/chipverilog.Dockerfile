FROM chipverilog-eval:ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/mage-chipverilog
RUN python3 -m pip install --break-system-packages /opt/mage-chipverilog

WORKDIR /opt/mage-chipverilog
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

