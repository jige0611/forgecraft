# ForgeCraft Docker 镜像 (CPU 版本)
FROM python:3.11-slim-bookworm

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先复制依赖与元数据 (利用 Docker 层缓存)
# 注: pyproject.toml 声明了 readme = "README.md", 构建元数据时需要 README/LICENSE
COPY pyproject.toml requirements.txt README.md LICENSE ./

# 安装核心依赖 (重型依赖利用缓存)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy mujoco gymnasium networkx pyyaml

# 复制源码并安装
COPY forgecraft/ forgecraft/
RUN pip install --no-cache-dir -e ".[dev]"

# 环境变量
ENV MUJOCO_GL=osmesa
ENV FORGECRAFT_BACKEND=mujoco
ENV PYTHONUNBUFFERED=1
ENV FORGECRAFT_VALIDATE=1

# 快速冒烟测试
RUN python -c "from forgecraft.config import EvolutionConfig; print('ForgeCraft OK')"

# 入口
ENTRYPOINT ["python", "-m", "forgecraft.main"]
CMD ["--help"]
