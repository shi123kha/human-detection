FROM python:3.8
EXPOSE 8080
# Install dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgtk2.0-dev pkg-config
RUN apt-get update && apt-get install -y \
    libv4l-dev \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . ./
ENTRYPOINT ["streamlit", "run", "StreamLiteApp.py", "--server.port=8080", "--server.address=0.0.0.0"]