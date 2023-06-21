FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y python3.10 \
    && apt-get install -y python3-pip \
    && apt-get install -y wget \
    && apt-get install -y unzip \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*


ARG PROJECT_PATH

ENV PROJECT_PATH=$PROJECT_PATH

RUN mkdir -p $PROJECT_PATH 

WORKDIR ${PROJECT_PATH}

# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED=1

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE=1

ENV PIP_DEFAULT_TIMEOUT=100

COPY ./requirements.dev.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.dev.txt
#WORKDIR /notebooks
ENV PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}

CMD ["/bin/bash"]