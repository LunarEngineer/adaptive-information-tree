FROM jupyter/datascience-notebook

RUN conda install --quiet --yes \
    'mlflow=1.0.0' \
    'psycopg2'
