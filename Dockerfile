FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir model raw_data processed_data results

# for preprocessing.py
ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV RAW_DATA_FILE=adult.csv
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV TRAIN_DATA_FILE=train.csv
ENV TEST_DATA_FILE=test.csv
# for train.py
ENV MODEL_DIR=/home/jovyan/model
ENV MODEL_FILE=model.joblib
ENV RESULTS_DIR=/home/jovyan/results
ENV TRAIN_RESULTS_FILE=train_metadata.json
# for test.py
ENV TEST_RESULTS_FILE=test_metadata.json
# for api.py

COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
COPY api.py ./api.py

RUN python3 preprocessing.py
RUN python3 train.py

