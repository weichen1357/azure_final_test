FROM --platform=linux/amd64 python:3.10-slim

RUN apt update

WORKDIR /app

COPY requirements.txt /app/
COPY web.py /app/
COPY templates /app/templates
COPY .env /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["python", "web.py"]