FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY bnp_app.py /app
COPY model /app/model/
COPY src /app/src/
CMD ["python", "bnp_app.py"]~