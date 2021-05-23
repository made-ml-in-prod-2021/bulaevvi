FROM python:3.8

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY model.pkl /model.pkl
COPY app.py /app.py

WORKDIR .

CMD ["python3", "app.py"]