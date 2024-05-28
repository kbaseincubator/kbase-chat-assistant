# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
    
# Copy Poetry files
COPY pyproject.toml poetry.lock ./
    
# Install dependencies
RUN poetry install 

# Copy the rest of the application code into the container
COPY . .

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Change the working directory to where app.py is located
WORKDIR /app/kbasechatassistant/user_interface

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["poetry", "run", "streamlit", "run", "app.py"]
