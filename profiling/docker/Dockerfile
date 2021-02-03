FROM python:3.8

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Add node
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -

RUN apt update && apt install -y libsnappy-dev openjdk-11-jdk build-essential apt-transport-https lsb-release ca-certificates curl nodejs

# Install Hadoop for hdfs
ENV HADOOP_VERSION 3.2.1
RUN curl -o - https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | tar xz
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV HADOOP_HOME /hadoop-${HADOOP_VERSION}
ENV PATH $HADOOP_HOME/bin:$PATH
ENV ARROW_LIBHDFS_DIR $HADOOP_HOME/lib/native
ENV CLASSPATH `$HADOOP_HOME/bin/hdfs classpath --glob`
COPY docker/hdfs-site.xml /hadoop-${HADOOP_VERSION}/etc/hadoop/hdfs-site.xml
COPY build/requirements.txt /workdir/requirements.txt

RUN pip install -r /workdir/requirements.txt
RUN pip install -U pyarrow python-snappy pandas-profiling distributed

WORKDIR /workdir

COPY docker/entrypoint.sh /usr/bin/entrypoint.sh
RUN chmod u+x /usr/bin/entrypoint.sh
ENTRYPOINT ["tini", "-g", "--", "/usr/bin/entrypoint.sh"]