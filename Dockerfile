# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PDF parsing by unstructured.io
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code into the container
COPY ./backend ./backend

# Copy Alembic configuration and migrations
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application
# Uvicorn will look for the 'app' instance in the 'main' module within the 'backend' directory
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
