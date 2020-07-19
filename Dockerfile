FROM jupyter/scipy-notebook

RUN pip install joblib

RUN mkdir model
ENV MODEL_DIR=/home/jovyan/model
#ENV MODEL_FILE=clf.joblib
#ENV METADATA_FILE=metadata.json

COPY train.py ./train.py
COPY data ./data
COPY inference.py ./inference.py
COPY requirements.txt ./requirements.txt
COPY instructions.pdf ./instructions.pdf
COPY Short_description.pdf ./Short_description.pdf
RUN pip3 install -r requirements.txt

RUN python3 train.py
