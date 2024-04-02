
FROM nvidia/cuda:11.5.2-cudnn8-devel-centos7


RUN yum install -y wget

RUN wget https://go.dev/dl/go1.22.1.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.22.1.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin
ENV GOPATH=/home/si/go
RUN rm go1.22.1.linux-amd64.tar.gz


RUN  echo "Installing Open MPI"
RUN  yum install -y perl
RUN  mkdir -p /opt/ompi
RUN  mkdir -p /tmp/ompi
RUN  wget  https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.gz
RUN  tar -C /tmp/ompi -xzf openmpi-5.0.0.tar.gz
RUN  cd /tmp/ompi/openmpi-5.0.0 && ./configure --prefix=/opt/ompi && make -j8 install
ENV  PATH=/opt/ompi/bin:$PATH
ENV  LD_LIBRARY_PATH=/opt/ompi/lib:$LD_LIBRARY_PATH
ENV  OMPI_ALLOW_RUN_AS_ROOT=1
ENV  OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV  C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/ompi/include/

RUN  cp -r /opt/ompi/lib/* /usr/local/lib/


# docker run --gpus all -v ~/go:/home/si/go -it 384b490e60c8

# docker images -a
# docker build . -t donego22mpi

# this is from docker image donego22mpi