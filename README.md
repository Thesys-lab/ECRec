# ECRec

## Overview

We present ECRec, a DLRM training system that achieves efficient fault tolerance by coupling erasure coding with the unique characteristics of DLRM training. ECRec takes a hybrid approach between erasure coding and replicating different DLRM parameters, correctly and efficiently updates redundant parameters, and enables training to proceed without pauses, while maintaining the consistency of the recovered parameters. We implement ECRec atop [XDL](https://github.com/alibaba/x-deeplearning), an open-source, industrial-scale DLRM training system. Compared to checkpointing, ECRec reduces training-time overhead on large DLRMs by up to 66%, recovers from failure up to 9.8× faster, and continues training during recovery with only a 7–13% drop in throughput (whereas checkpointing must pause).

## Quick Links

ECRec is implemented atop XDL and therefore its setup is highly similar to that of XDL and its interface is fully consistent with XDL. For details, read more at XDL's official documentation. Links below have routed through Google Translate as XDL docs was written in Chinese.

**You do not need to go through the installation steps in the link if you opt to use our Docker image. We provide the link for your reference in case you encounter problems following our setup steps.**

* [Compilation & Installation](https://github-com.translate.goog/alibaba/x-deeplearning/wiki/%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp)
* [Getting Started](https://github-com.translate.goog/alibaba/x-deeplearning/wiki/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp)
* [Usage](https://github-com.translate.goog/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp)

## Manual Run (Development)
### Installation

We have prepared a Docker image [`kaigel1998/xdl_installed:v3`](https://hub.docker.com/layers/kaigel1998/xdl_installed/v3/images/sha256-553030a64043b89f572812f4ab678527d7cdd3c7b2c2b8ccc5adfd03214b562a?context=explore) hosted on Docker Hub that contains the necessary environment to run ECRec. We provide some example commands for your reference to get started.

```sh
sudo apt-get update && sudo apt-get -y install docker.io
sudo systemctl start docker
sudo docker pull kaigel1998/xdl_installed:v3
sudo docker run -it --network=host kaigel1998/xdl_installed:v3
apt update && apt install vim -y
cd /x-deeplearning-redundancy/xdl/build/
git remote remove origin
git remote add origin https://github.com/Thesys-lab/ECRec.git
git config --global credential.helper store
echo "<your github token>" > ~/.git-credentials
git reset --hard HEAD^ && git pull
git checkout ecrec
cmake .. -DTF_BACKEND=1 && make -j$(nproc) && make install_python_lib
cd ../examples/criteo
```

### Compilation & Running

After you make code changes, run the following to test out
```sh
cd /x-deeplearning-redundancy/xdl/build/
cmake .. -DTF_BACKEND=1 && make -j$(nproc) && make install_python_lib
cd ../examples/criteo # or your own path where .py launchers reside
```

You will need to spawn a scheduler, at least one parameter server (PS), and at least one worker for training to begin. We provide example launching commands below. These commands launch the necessary ECRec instances on a single host. Note that our experiments need the Criteo Terabyte dataset downloaded to a local path on the worker machine. The following command downloads a part of the pre-processed dataset from our server. All processed parts of the Criteo Terabyte dataset can be found [here](https://ftp.pdl.cmu.edu/pub/datasets/DLRM/criteo-terabytes/).

```sh
# scheduler
apt-get update && apt-get install -y zookeeper  \
&& /usr/share/zookeeper/bin/zkServer.sh stop  \
&& /usr/share/zookeeper/bin/zkServer.sh start  \
&& /usr/share/zookeeper/bin/zkCli.sh create /scheduler 'scheduler'  \
&& /usr/share/zookeeper/bin/zkCli.sh get /scheduler \
&& python criteo_training.py --task_name=scheduler --zk_addr=zfs://localhost:2181/scheduler --ps_num=1 --ps_cpu_cores=6 --ps_memory_m=64000 --ckpt_dir=.

# ps
python criteo_training.py --task_name=ps --zk_addr=zfs://0.0.0.0:2181/scheduler --task_index=0

# worker
mkdir /xdl_training_samples && wget https://ftp.pdl.cmu.edu/pub/datasets/DLRM/criteo-terabytes/day_0_processed_tiny_0 -O /xdl_training_samples/data.txt

python criteo_training.py --task_name=worker --zk_addr=zfs://0.0.0.0:2181/scheduler --task_index=0 --task_num=1
```

Note that you may need to change/tune parameters in the above commands to obtain decent performance.

## Bulk Run (Experiments)

XDL/ECRec is designed to run distributedly on a set of hosts over a network. To enable repeatable and reproducible experiments, we provide a reference experiment launching program that allows spawning ECRec clusters on AWS and training on them with simple commands. The program can be found in this repo at [`launch_exp.py`](launch_exp.py). You will need to fill in AWS EC2 keypair and GitHub credentials information in the program script. Additional useful variables to modify are described below.

Common usage includes:

* Spawn cluster: `python init <branch> <num_workers>`. This launches and initializes AWS EC2 instances with the environment and docker images.
    - The `INSTANCE_SPECS` variable in the script specifies the number of servers/workers and corresponding EC2 instance types, following the current tuple format.
    - `S3_FILES`: an array of files that are used as training data. Each file corresponds to one worker, so the array size must >= the number of workers.
    - `PS_MEMORY_MB`: server memory restriction.
    - `PS_NUM_CORES`: server number of cores.
* Launch experiments: `python run <branch> <num_workers>`. This kills existing docker containers and runs all scheduler, servers and workers.
    - `TRAINING_FILE`: which python file in the directory to use for training. 
* Run workers only: `python run_workers_only <branch> <num_workers>`. This kills existing docker containers of workers and run workers, keeping the scheduler and servers alive. 
    - This is useful when conducting recovery experiments, where you would want to SSH into servers and run workers with this script.

To trigger recovery, SSH into a PS host and kill and rerun its docker image.

Throughput metrics will be logged into the path specified by the `OUTPUT_DIR` variable in the experiment launching program. Refer to line 15 of [`criteo_training.py`](xdl/examples/criteo/criteo_training.py) to understand the numbers in the logged tuple. We provide a simple script [`parse_results.py`](parse_results.py) to aggregate the throughput metrics across all hosts for your reference.

## Reproducing Results in Paper

We ran a multitude of experiments to obtain results presented in the paper. Given the intricate nature of fault tolerance in distributed training, we regret that we cannot offer an end-to-end script to execute all experiments and generate the results. However, to ensure reproducibility, we have provided a comprehensive [list of configurations](EXP_CONFIGS.md) used in our experiments. 

To replicate our results, please refer to the [Bulk Run (Experiments)](#bulk-run-experiments) section and perform a bulk run for each specified configuration. By following these instructions, you should be able to obtain results consistent with those reported in the paper.

## Support
We graciously acknowledge support from the National Science Foundation (NSF) in the form of a Graduate Research Fellowship (DGE-1745016 and
DGE-1252522), in part by a TCS Presidential Fellowship, Amazon Web Services, a VMware Systems Research Award, and the AIDA project (POCI-01-0247- FEDER-045907) co-financed by the European Regional Development Fund through the Operational Program for Competitiveness and Internationalisation 2020.  We also thank [CloudLab]([url](https://www.cloudlab.us/)) for providing computational resources used in carrying out part of this research.
