FROM python:3.8

# Install libsundials
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libsundials-dev

# Install dependencies
RUN pip install --upgrade pip
COPY ./ /erlotinib
RUN cd /erlotinib && pip install --no-cache-dir -r requirements.txt

WORKDIR /erlotinib

EXPOSE 8050
CMD ["python", "./erlotinib/apps/_simulation.py"]