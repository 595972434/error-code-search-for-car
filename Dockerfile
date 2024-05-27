FROM python:3.11.9

WORKDIR /app


COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY data /app/data
COPY model /app/model
COPY main.py /app/main.py

EXPOSE 8501

CMD [ "streamlit", "run", "main.py"]