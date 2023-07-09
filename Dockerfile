FROM pytorch/pytorch:latest

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

#RUN mkdir output

COPY unet.py .
COPY network.py .
COPY measure.py .
COPY train.py .
CMD ["python", "train.py"]