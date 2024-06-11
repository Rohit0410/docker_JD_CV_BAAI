FROM python:3.9-slim-buster
WORKDIR /api
COPY ./requirements.txt /api
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5005
ENV FLASK_APP=api.py
CMD ["flask", "run", "--host=0.0.0.0","--port=5005"]