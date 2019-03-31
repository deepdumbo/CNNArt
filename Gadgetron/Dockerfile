FROM gadgetron/ubuntu_1804_cuda92_cudnn7:latest

# RUN buildDeps='gcc libc6-dev make wget' \
#     && apt-get update \
#     && apt-get install -y $buildDeps \
#     && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
#     && mkdir -p /usr/src/redis \
#     && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
#     && make -C /usr/src/redis \
#     && make -C /usr/src/redis install \
#     && rm -rf /var/lib/apt/lists/* \
#     && rm redis.tar.gz \
#     && rm -r /usr/src/redis \
#     && apt-get purge -y --auto-remove $buildDeps

RUN pip install pydicom dicom-numpy scikit-learn pyyaml h5py \
    # && wget -d https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    # && ./Miniconda3-latest-Linux-x86_64.sh \
    # && conda install theano pygpu -y \
    # && conda install -c clinicalgraphics gdcm \
    && cd /opt \ 
    && git clone https://github.com/thomaskuestner/CNNArt.git \
    && git checkout motion_correction_Yang_Liu \
    && mkdir /opt/data \
    && cp /opt/CNNArt/Gadgetron/cnnart.xml /usr/local/share/gadgetron/config

COPY ../../cnnart_trainednets /opt/CNNArt
WORKDIR /opt/data
