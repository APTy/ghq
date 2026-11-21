# Build from the repository root, for example:
# docker buildx build --platform linux/amd64 -f training/Dockerfile -t ghq-train:pilot --load .
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-neural.txt /app/requirements-neural.txt
# Preserve the CUDA-enabled torch supplied by the base image; fail on conflicts.
RUN printf 'torch==2.7.1\n' > /tmp/torch-constraint.txt \
    && python -m pip install --no-cache-dir -c /tmp/torch-constraint.txt -r requirements-neural.txt
COPY ghqnet /app/ghqnet
COPY public/engine.py /app/public/engine.py
COPY public/search.py /app/public/search.py
ENTRYPOINT ["python", "-m", "ghqnet.pipeline"]
