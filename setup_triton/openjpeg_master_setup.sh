git clone https://github.com/uclouvain/openjpeg.git
cd openjpeg/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
make clean

