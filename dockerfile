FROM nvcr.io/nvidia/pytorch:22.11-py3 

COPY ./setup_base/ /setup_base/

RUN bash /setup_base/openjpeg_master_setup.sh

RUN bash /setup_base/apt_updates.sh

