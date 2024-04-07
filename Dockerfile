
FROM python:3.8-slim

WORKDIR /app

# Installation des packages Python nécessaires
RUN pip install pandas numpy scikit-learn flask gunicorn

COPY app.py .
COPY templates/ templates/


COPY iris_model.pkl .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
