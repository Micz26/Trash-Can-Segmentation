# Base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files first to leverage Docker cache
COPY pyproject.toml .
COPY src ./src

# Install the app and its dependencies
RUN pip install --no-cache-dir .

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/trashcan_frontend/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
