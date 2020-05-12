
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

### User setting ### default is "daisuke"
ARG NB_USER=kazuya
ENV NB_USER ${NB_USER}
# HOST_ID should be same as host's user id.
ARG HOST_UID
ENV NB_UID ${HOST_UID}

# Add user and set password.
RUN useradd -m -G sudo -s /bin/bash -U $NB_UID $NB_USER && \
    echo "${NB_USER}:password" | chpasswd

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo 'kazuya ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN apt-get update && apt-get install -y rsync htop git openssh-server

RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
RUN pip install torchvision cffi tensorboardX
RUN pip install tqdm scipy scikit-image colorama==0.3.7 
RUN pip --no-cache-dir install ipykernel jupyter && python -m ipykernel.kernelspec
RUN pip install setproctitle pytz ipython
RUN pip install --no-cache ptvsd

RUN git clone https://github.com/facebookresearch/visdom.git /root/visdom
RUN cd /root/visdom && \
    git checkout $(cat /root/commitish)
LABEL io.k8s.description="Visdom server" \
      io.k8s.display-name="Visdom server"

ADD ./commitish /root/
ADD ./visdom_connection_test.py /root/

RUN cd /root/visdom && \
    git checkout $(cat /root/commitish)

LABEL io.k8s.description="Visdom server" \
      io.k8s.display-name="Visdom server"

ENV HOSTNAME='localhost'
ENV PORT=8097
ENV ENV_PATH="~/.visdom/"
ENV LOGGING_LEVEL='INFO'
ENV READONLY="True"
ENV ENABLE_LOGIN="False"
ENV FORCE_NEW_COOKIE="False"
ENV BASE_URL="/"

RUN cd /root/visdom && pip install .

EXPOSE $PORT

EXPOSE 8888

COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

CMD python -m visdom.server \
    --hostname ${HOSTNAME} \
    -port ${PORT} \
    -base_url ${BASE_URL} \
    -env_path ${ENV_PATH} \
    -logging_level ${LOGGING_LEVEL} \
    `if [ "x$READONLY" = "xTrue" ];then echo "-readonly";fi` \
    `if [ "x$FORCE_NEW_COOKIE" = "xTrue" ];then echo "-force_new_cookie";fi` \
    `if [ "x$ENABLE_LOGIN" = "xTrue" ];then echo "-enable_login";fi`

WORKDIR "/notebooks"
CMD ["/run_jupyter.sh", "--allow-root"]

