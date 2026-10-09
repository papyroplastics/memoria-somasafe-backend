FROM docker.io/library/debian:trixie-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project --group bench

FROM docker.io/library/debian:trixie-slim
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
COPY common common
COPY benchmark benchmark
ENTRYPOINT ["python", "-m"]
CMD ["benchmark.run", "--help"]
