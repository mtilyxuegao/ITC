#!/bin/bash
# Bring up the PyTorch MiniCPM-o demo on B300 with torch.compile enabled.
# Single GPU (0), port 8011, project minicpm-pt. Avoids the llama.cpp Comni stack (8006).
set -e
cd /root/MiniCPM-o-Demo

# 1) Derived worker image: base + gcc (Triton needs a C compiler) + sm_103a ptxas + soundfile patch
mkdir -p /root/ptbuild
cp /root/ptxas13 /root/ptbuild/ptxas13
cp /root/sitecustomize.py /root/ptbuild/sitecustomize.py
cat > /root/ptbuild/Dockerfile <<'DOCKER'
FROM minicpm-wb-b300:dev
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*
COPY ptxas13 /usr/local/bin/ptxas13
COPY sitecustomize.py /app/sitecustomize.py
ENV TRITON_PTXAS_PATH=/usr/local/bin/ptxas13 \
    CC=gcc \
    TORCHINDUCTOR_CACHE_DIR=/app/torch_compile_cache \
    PYTHONPATH=/app
DOCKER
echo "[1/4] building minicpm-wb-b300-compile:dev ..."
docker build -q -t minicpm-wb-b300-compile:dev /root/ptbuild

# 2) config.json with compile=true
python3 - <<'PY'
import json
c = json.load(open('/root/MiniCPM-o-Demo/config.example.json'))
# set compile=true wherever it lives (nested under service or flat)
def setc(d):
    for k,v in d.items():
        if k=='compile': d[k]=True
        elif isinstance(v,dict): setc(v)
setc(c)
json.dump(c, open('/root/config_compile.json','w'), indent=2, ensure_ascii=False)
print('compile flag set:', json.dumps(c).count('"compile": true') or 'check')
PY

# 3) single-worker compose
cat > /root/MiniCPM-o-Demo/docker-compose.b300pt.yml <<'YML'
services:
  worker-backend-0:
    image: minicpm-wb-b300-compile:dev
    environment:
      GPU_ID: "0"
      BACKEND_PORT: "22500"
      WORKER_PORT: "22400"
      MODEL_PATH: /models/MiniCPM-o-4_5
    volumes:
      - /root/models/MiniCPM-o-4_5:/models/MiniCPM-o-4_5:ro
      - /root/config_compile.json:/app/config.json:ro
      - /root/torch_compile_cache:/app/torch_compile_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD","curl","-sf","http://127.0.0.1:22400/health"]
      interval: 15s
      timeout: 5s
      retries: 60
      start_period: 240s
    restart: "no"
  gateway:
    image: minicpm-gateway:dev
    command: ["--host","0.0.0.0","--port","8006","--https","--ssl-certfile","/app/certs/cert.pem","--ssl-keyfile","/app/certs/key.pem","--workers","worker-backend-0:22400"]
    ports:
      - "8011:8006"
    volumes:
      - ./data:/app/data
      - ./certs:/app/certs:ro
    depends_on:
      worker-backend-0:
        condition: service_healthy
    restart: "no"
YML

# 4) up
echo "[4/4] docker compose up ..."
docker compose -p minicpm-pt -f docker-compose.b300pt.yml up -d
docker compose -p minicpm-pt -f docker-compose.b300pt.yml ps
