FROM pytorch/pytorch:latest

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

COPY unet.py .
COPY network.py .
COPY measure.py .
COPY train.py .
ENTRYPOINT ["python", "train.py"]