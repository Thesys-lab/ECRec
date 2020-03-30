# Building XDL on a Ubuntu machine (CPU version only)
Start with a Ubuntu machine, with a directory (such that /mydata) mounted with 128GB at least.

Install docker and set the default path to "/mydata"
```
sudo apt-get update
sudo apt-get install docker.io
sudo vi /etc/docker/daemon.json
```
Add the line ```{ "graph":"/mydata" }``` in /etc/docker/daemon.json, and restart docker
```
sudo systemctl stop docker
sudo systemctl start docker
```

## Option 1: BUILD FROM SCRATCH
Install system updates
```
sudo apt-get update
sudo apt-get install docker.io
```

Pull and run docker image
```
sudo docker pull ubuntu:16.04
sudo docker run --net=host -it ubuntu:16.04 /bin/bash
```

Now we are in the docker image. Install system updates:
```
apt-get update && apt-get -y upgrade
apt-get install -y build-essential gcc-4.8 g++-4.8 gcc-5 g++-5 cmake python python-pip openjdk-8-jdk wget && pip install --upgrade pip && apt-get install -y libaio-dev ninja-build ragel libhwloc-dev libnuma-dev libpciaccess-dev libcrypto++-dev libxml2-dev xfslibs-dev libgnutls28-dev liblz4-dev libsctp-dev libprotobuf-dev protobuf-compiler libunwind8-dev systemtap-sdt-dev libjemalloc-dev libtool python3 libjsoncpp-dev apt-transport-https curl git zip python-dev python-pip
```
Install boost
```
cd /tmp
wget -O boost_1_63_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz
tar zxf boost_1_63_0.tar.gz && cd boost_1_63_0
./bootstrap.sh --prefix=/usr/local && ./b2 -j32 variant=release define=_GLIBCXX_USE_CXX11_ABI=0 install
mkdir -p /usr/local/lib64/boost && cp -r /usr/local/lib/libboost* /usr/local/lib64/boost/
```
Prepare Tensorflow dependencies
```
pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U --user keras_applications --no-deps
pip install -U --user keras_preprocessing --no-deps
pip install futures enum34
```
Install bazel-0.15.2
```
cd /
wget https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-installer-linux-x86_64.sh
chmod +x bazel-0.15.2-installer-linux-x86_64.sh
./bazel-0.15.2-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
```
Install Tensorflow r1.12
```
cd /
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.12
./configure
bazel build --config opt //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.12.3-cp27-cp27mu-linux_x86_64.whl
```
Install xdl
```
cd /
git clone --recursive https://github.com/DUKaige/x-deeplearning.git
cd x-deeplearning/xdl
mkdir build && cd build
export CC=/usr/bin/gcc-5 && export CXX=/usr/bin/g++-5
cmake .. -DTF_BACKEND=1
make -j64 
make install_python_lib
```
Test it out!
```
pip install matplotlib==2.0.0 protobuf==3.6.1 pyhdfs
python -c "import xdl; print xdl.__version__"
```
Commit docker image
```
sudo docker ps -l
sudo docker commit <container_id> <image>
```

# Option 2: Download my docker image directly
```
docker pull kaigel1998/xdl_installed:v2
```

# Trouble shooting
Tensorflow operator error: reinstall Tensorflow following the steps above.

# Run distributedly
## Create ZK
```
apt-get install zookeeper
/usr/share/zookeeper/bin/zkServer.sh start
/usr/share/zookeeper/bin/zkCli.sh create /scheduler "scheduler"
/usr/share/zookeeper/bin/zkCli.sh get /scheduler
```
## Run scheduler
```
python deepctr.py --task_name=scheduler --zk_addr=zfs://localhost:2181/scheduler --ps_num=1 --ps_cpu_cores=2 --ps_memory_m=1000 --ckpt_dir=. >schedule.log 2>&1 &
```
## Run Server
```
python deepctr.py --task_name=ps --task_index=0 --zk_addr=zfs://localhost:2181/scheduler >ps.log 2>&1 &
```
## Run Clients 
```
python deepctr.py --task_name=worker --task_index=0 --task_num=2 --zk_addr=zfs://localhost:2181/scheduler >worker0.log 2>&1 &
python deepctr.py --task_name=worker --task_index=1 --task_num=2 --zk_addr=zfs://localhost:2181/scheduler >worker1.log 2>&1 &  
```
