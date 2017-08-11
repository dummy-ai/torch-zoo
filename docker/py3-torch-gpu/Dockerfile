FROM py3-numpy-gpu

MAINTAINER Jim Fan <jimfan@cs.stanford.edu>


RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install torchvision

# create symlinks
RUN ln -fs /usr/bin/python3 /usr/bin/python \
    && ln -fs /usr/bin/pip3 /usr/bin/pip

ENTRYPOINT ["/core/init"]
