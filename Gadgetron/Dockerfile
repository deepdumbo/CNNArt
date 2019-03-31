# $ docker build -t cnnart_gadgetron .
FROM gadgetron/ubuntu_1804_cuda92_cudnn7:latest

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 \
    && pip install --upgrade pip \
    && pip install --upgrade pydicom dicom-numpy scikit-learn pyyaml h5py tensorflow keras matplotlib\
    # && wget -d https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    # && ./Miniconda3-latest-Linux-x86_64.sh \
    # && conda install theano pygpu -y \
    # && conda install -c clinicalgraphics gdcm \
    && cd /opt \ 
    && git clone https://github.com/thomaskuestner/CNNArt.git \
    && cd CNNArt \
    && git checkout motion_correct_Yang_Liu \
    && mkdir /opt/data \
    && cp /opt/CNNArt/Gadgetron/cnnart.xml /usr/local/share/gadgetron/config

# manual move cnnart_trainednets to /opt/CNNArt
WORKDIR /opt/data

# cd to work direction
# $ docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron
# $ docker exec -ti gt1 /bin/bash
# in docker: $ gadgetron_ismrmrd_client -f testdata.h5 -o out.h5 -c cnnart.xml

# next run:
# $ docker start gt1
# $ docker exec -ti gt1 bash

