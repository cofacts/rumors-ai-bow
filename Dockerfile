FROM python:3.8

EXPOSE 80

ADD requirements.txt .
RUN python -m pip install -r requirements.txt

CMD ["python", "main.py"]
