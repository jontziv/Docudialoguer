FROM python:3.11.5

WORKDIR /app

RUN apt-get update \
    && apt-get install -y  --no-install-recommends \
        apt-utils \
        locales \
        python3-pip \
        python3-yaml \
        rsyslog systemd systemd-cron sudo\
    && apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


EXPOSE 11434
CMD ["chainlit", "run", "agent.py", "--port", "7860"]