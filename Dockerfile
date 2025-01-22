FROM python:3.9-slim

WORKDIR /app

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean

# Copy all application files into the container
COPY . . 

# Install pip dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Expose the necessary port and run the app
EXPOSE 8501
CMD ["streamlit", "run", "src/trashcan_frontend/frontend.py", "--server.address=0.0.0.0"]
