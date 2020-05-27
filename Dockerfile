FROM pytorch/pytorch

RUN apt-get update && \
  apt install -y tmux && \
  apt install -y openssh-server && \
  service ssh start

RUN pip install torchaudio gpustat ffmpeg-python opencv-python
RUN mkdir lib && cd lib
RUN git clone https://github.com/alexandonian/pretorched-x.git && cd pretorched-x && pip install -e . && git checkout dev && cd ..
RUN git clone https://github.com/alexandonian/torchvideo.git && cd torchvideo && pip install -e . && cd ..
RUN git clone https://github.com/alexandonian/lintel.git && cd lintel && pip install -e . && cd ..
RUN git clone https://github.com/alexandonian/FileLock.git && cd FileLock && pip install -e . && cd ..
