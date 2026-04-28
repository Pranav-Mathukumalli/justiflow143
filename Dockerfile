FROM python:3.10

WORKDIR /app

# Tell the container which port to use
ENV PORT=8080

COPY . .

RUN pip install -r requirements.txt

# Open the port for external traffic
EXPOSE 8080

CMD ["python", "app.py"]
