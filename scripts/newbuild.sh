mv build/_deps .
rm -r build
mkdir build
mv _deps build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 20
