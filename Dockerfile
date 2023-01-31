# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN apt-get install -y libsndfile1

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD setup.py .
CMD python3 setup.py install

# We add the banana boilerplate here
ADD . .

RUN python3 download.py

EXPOSE 8000

CMD python3 -u server.py