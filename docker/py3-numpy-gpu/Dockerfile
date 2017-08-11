FROM dummyai/python3-gpu

RUN apt update \
    && apt install -y sox libsox-fmt-all protobuf-compiler python-pil python-lxml tmux htop
    
RUN pip3 install numpy scipy jupyter matplotlib Pillow seaborn tabulate pandas
RUN pip3 install pysftp paramiko flask bs4
RUN pip3 install librosa>=0.5 pyxdg python_speech_features sox

# create symlinks
RUN ln -fs /usr/bin/python3 /usr/bin/python \
    && ln -fs /usr/bin/pip3 /usr/bin/pip

# COPY core /core

ENTRYPOINT ["/core/init"]
