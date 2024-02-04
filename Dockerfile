# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install virtualenv
RUN pip install virtualenv

# Create and activate a virtual environment
RUN virtualenv .venv
RUN /bin/bash -c "source .venv/bin/activate"

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the project files into the container
COPY project project

# Run the ml_main script
CMD ["python", "-m", "project.ml_model.ml_main"]

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--reload"]
