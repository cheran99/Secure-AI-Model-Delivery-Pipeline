FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Install the application dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ .
COPY model.joblib .

# run as non-root
USER appuser

# Expose the port your app will be on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]