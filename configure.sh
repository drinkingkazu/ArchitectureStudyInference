#!/bin/bash

export HBD_CHURCH_DIR=/data/shared/ArchitectureStudyInference
#export HBD_CHURCH_DIR=/data/shared/sp_shared_ana;
#export HBD_CHURCH_DIR=/data/drinkingkazu/SPClassification/py_inference
mkdir -p $HBD_CHURCH_DIR;
chmod 775 -R $HBD_CHURCH_DIR;

if [[ -d $HBD_CHURCH_DIR/larcv ]]; then
    source $HBD_CHURCH_DIR/larcv/configure.sh;
else
    git clone https://github.com/LArbys/LArCV $HBD_CHURCH_DIR/larcv;
    cd $HBD_CHURCH_DIR/larcv;
    git checkout spatial_weigt;
    source configure.sh;
    make -j6;
    cd -;
fi


if [[ -d $HBD_CHURCH_DIR/caffe ]]; then
    source $HBD_CHURCH_DIR/caffe/configure.sh;
else
    cd $HBD_CHURCH_DIR;
    git clone https://github.com/LArbys/caffe $HBD_CHURCH_DIR/caffe;
    cd $HBD_CHURCH_DIR/caffe;
    git checkout more_io_thread;
    ln -s Makefile.config.wu Makefile.config;
    source configure.sh;
    make -j6;
    make pycaffe;
    cd -;
fi
