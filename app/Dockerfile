FROM bhyveregistry.azurecr.io/bhyve-devops-cuda-llm-deps:v1.0.0

WORKDIR /app

COPY ./app /app
COPY ./entrypoint.sh /entrypoint.sh

USER root

RUN apt-get update \
  && apt-get install -y \
  nano \
  && chmod +X /entrypoint.sh

ENTRYPOINT tail -f /dev/null &wait

CMD ["python3", "app.py"]
