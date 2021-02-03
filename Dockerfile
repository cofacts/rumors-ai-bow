FROM python:latest

# ADD requirements.txt .
# ADD main.py .
# COPY model .

WORKDIR /app

ADD . /app
RUN python -m pip install -r requirements.txt

CMD ["python", "main.py"]
