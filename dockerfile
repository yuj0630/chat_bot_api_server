FROM python:3.12.9

WORKDIR /code

# Install system dependencies for mysqlclient and pkg-config
COPY . /code/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Expose the port the app runs on
EXPOSE 8892

# Define environment variable for Flask
ENV FASTAPI_APP=main.py
ENV FASTAPI_RUN_HOST=0.0.0.0

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8892", "--reload"]