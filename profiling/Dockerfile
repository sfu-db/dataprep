FROM python:3.8

WORKDIR /workdir

RUN apt update && apt install -y libsnappy-dev

RUN apt install --assume-yes build-essential apt-transport-https lsb-release ca-certificates curl

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -

RUN apt install --assume-yes nodejs

COPY build/requirements.txt /workdir

RUN pip install -r /workdir/requirements.txt

RUN pip install -U pyarrow python-snappy pandas-profiling

RUN jupyter contrib nbextension install
RUN jupyter nbextensions_configurator enable
RUN jupyter labextension install dask-labextension
RUN jupyter serverextension enable dask_labextension