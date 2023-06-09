# Use an official Python runtime as a parent image
FROM python:3.11.3-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Move the code
COPY . /app

ENV PYTHONPATH=/app/src
# Set the entry point
ENTRYPOINT ["python", "src/linear_regression/linear_regression.py"]
