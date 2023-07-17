""" Utility script for launching EC2 instances """
import boto3
import time
import threading
from subprocess import Popen
import argparse
import os

# TODO: add aws copying files: copy s3 file to local disk, add docker mapping, and add criteo_training argument.
cpu_ami = "ami-0d99f7305844e7cf4"
#cpu_ami = "ami-83f277e3"
gpu_ami = "ami-0e6e6f43a88422048"

security_groups = ["ParmSecurityGroup"]
target_availability_zone = "us-west-2b"
target_placement_group = 'xdl-cluster'
PUBLIC_IP_ADDR_PATH = './public_ip_addresses'
PRIVATE_IP_ADDR_PATH = './private_ip_addresses'
HDFS_PUBLIC_IP_ADDR_PATH = './hdfs_public_ip_addresses'
HDFS_PRIVATE_IP_ADDR_PATH = './hdfs_private_ip_addresses'
ZOOKEEER_SETUP_SCRIPT = "apt-get update && apt-get install -y zookeeper " \
                        "&& /usr/share/zookeeper/bin/zkServer.sh stop " \
                        "&& /usr/share/zookeeper/bin/zkServer.sh start " \
                        "&& /usr/share/zookeeper/bin/zkCli.sh create /scheduler 'scheduler' " \
                        "&& /usr/share/zookeeper/bin/zkCli.sh get /scheduler && "


CORE_SITE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
    <configuration>
        <property>
            <name>fs.default.name</name>
            <value>hdfs://{ip}:9000</value>
        </property>
    </configuration>
"""

YARN_SITE_TEMPLATE = """<configuration>
    <property>
            <name>yarn.acl.enable</name>
            <value>0</value>
    </property>

    <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>{ip}</value>
    </property>

    <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
    </property>
</configuration>

"""
SERVER_DATA_PATH = "/xdl_training_samples/data.txt"
CPU_DOCKER_IMG = "kaigel1998/xdl_installed:v3"
GPU_DOCKER_IMAGE = "kaigel1998/xdl_gpu:v5"

# ----------------------------------------------------------------------------------------------------------------------
# this section are the experiment configs
# INSTANCE_SPECS = [
#     ('scheduler', 1, 'cpu', 'r5n.4xlarge'), 
#     ('ps', 4, 'cpu', 'r5n.8xlarge'),
#     ('worker', 1, 'cpu', 'c5.2xlarge'), 
#     # ('worker', 2, 'gpu', 'p3.2xlarge')
# ]

# realtime ckpt exp
# INSTANCE_SPECS = [
#     ('scheduler', 1, 'cpu', 'r5n.xlarge'), 
#     ('ps', 1, 'cpu', 'r5n.2xlarge'),
#     # ('worker', 1, 'cpu', 'c5.2xlarge'), 
#     ('worker', 1, 'gpu', 'p3.2xlarge')
# ]
# PS_MEMORY_MB = 64000 #160000
# PS_NUM_CORES = 8 #16


# paper setup
INSTANCE_SPECS = [
    ('scheduler', 1, 'cpu', 'r5n.8xlarge'), 
    ('ps', 5, 'cpu', 'r5n.24xlarge'), #'r5n.8xlarge'
    ('worker', 15, 'gpu', 'p3.2xlarge')
]
PS_MEMORY_MB = 760000 #250000
PS_NUM_CORES = 96 #32

# INSTANCE_SPECS = [
#     ('scheduler', 1, 'cpu', 'r5n.8xlarge'),
#     ('ps', 5, 'cpu', 'r5n.8xlarge'),
#     ('worker', 2, 'cpu', 'c5.4xlarge'), #c5.9xlarge
#     # ('worker', 2, 'gpu', 'p3.2xlarge')
# ]
HDFS_CLUSTER_SIZE = 2
HDFS_CLUSTER_INSTANCE_TYPE = "i3en.xlarge"

# ----------------------------------------------------------------------------------------------------------------------
# other experiment configs'
# S3_FILES = ['day_0_processed_tiny_0', 'day_0_processed_tiny_1', 'day_0_processed_tiny_2', 'day_0_processed_tiny_3', 'day_0_processed_tiny_4', 'day_0_processed_tiny_5', 'day_0_processed_tiny_6', 'day_0_processed_tiny_7', 'day_0_processed_tiny_8', 'day_0_processed_tiny_9', 'day_0_processed_tiny10'] * 5
S3_FILES = ['day_0_processed_mini_0', 'day_0_processed_mini_1','day_0_processed_mini_2','day_0_processed_mini_3','day_0_processed_mini_4','day_0_processed_mini_5','day_0_processed_mini_6','day_0_processed_mini_7','day_0_processed_mini_8','day_0_processed_mini_9','day_0_processed_mini_10','day_0_processed_mini_11','day_0_processed_mini_12','day_0_processed_mini_13','day_0_processed_mini_14','day_0_processed_mini_15','day_0_processed_mini_16','day_0_processed_mini_17','day_0_processed_mini_18','day_0_processed_mini_19'] * 5
# S3_FILES = ['day_0_processed_mini_0'] * 100
# EXPERIMENT_BRANCH = "redundancy"
EXPERIMENT_BRANCH = "mlc"
TRAINING_FILE = "criteo_training.py"

EBS_VOLUME = 500
LAUNCH_TO_SSH_SLEEP_DURATION = 300
# PS_MEMORY_MB = 64000 #160000
# PS_NUM_CORES = 8 #16
SCHEDULER_DELAY_DURATION = 30
NUM_REPEAT = 1
NUM_WORKER_TASKS_PER_INSTANCE = 1

#CKPT_DIR="hdfs://{ip}:9000/"
CKPT_DIR="."

KEYPAIR_NAME = None # <KEY_PAIR_NAME_ON_AWS> # example: 'Tianyu_Oregon'
KEYPAIR_PATH = None # <PATH_TO_KEY_FILE> # example: '~/aws/Tianyu_Oregon.pem'

GITHUB_USERNAME = None
GITHUB_TOKEN = None

if KEYPAIR_NAME or KEYPAIR_PATH or GITHUB_USERNAME or GITHUB_TOKEN is None:
    raise ValueError("Need to fill in AWS EC2 keypair and GitHub credentials.")
# ----------------------------------------------------------------------------------------------------------------------
# send instruction script
#INSTRUCTION_TO_SEND = "sudo docker stop \$(sudo docker ps -aq) && sudo docker rm \$(sudo docker ps -aq)"
INSTRUCTION_TO_SEND = "echo \"ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCuzhQnhNLW4Cj4vi7kqJB+vLVs0+1z/QTU/75MsZZwU/ivDqsFydU9A+PpH7CIXzecJ4jDl7tBaDMT6fvaLxxPFsOZyTi3aINwWgsGYpyWW5tEDAj5In4GyRUhiie9G9GMwnSa45XgXzszPYMct38OL/jWLaVBlIPEwTn0TeJGCltP9AbuiB6PVuSaerx7GUrBWPjWMMn3Tzop/2ZQCkHC0Be0yyxaVJ8axnHjHiTD+rJkpkUaV8PoWEZJ0NtH0X+RA30x0t9sYriA5QbB6yZRlX83BezthdqFL+ND9+n4MpC6osJSKxzdshqQOuXXKkEQGZ0xKwVoitNEQJBKgfMc99ftYMYl78ET8BFz5o8g86qvtFmWuUQfIpzJF4293qGFuMt3FjIGqIxUyMQXLhCCoiRR+nRhpW7EE5pmOOCP8zFoufa5db/Ovau/59pF1A5yMdw4foiNtKR2G/jfIMoT8WdwiAuUyv6uIpoFYCv8jywT1I1DLl/d+n3VdqAXRS1qk+mCieEs6a2qXlsJvTtb5u+dnB7OKCw64THpH5JguxWJ9A1Th8VgdRmbfGiJOlLLRwLxD+W6HQdFziK2DK6PFiazAAQrPgnUcItLicSUSsChwQCnPqOSkP2/+KXZ2CgXbd2Jdw6OxEUPPRKDXyEZQ2LbbFWEUSUlVe57Xu/pEw== ubuntu@ip-172-31-24-111\" >> ~/.ssh/master.pub && cat ~/.ssh/master.pub >> ~/.ssh/authorized_keys"
#INSTRUCTION_TO_SE
# ND = "sudo file -s /dev/nvme1n1 " \
#              "&& sudo mOkkfs -t xfs /dev/nvme1n1 && sudo mkdir /mydata && sudo mount /dev/nvme1n1 /mydata " \
#              "&& sudo chown -R ubuntu:root /mydata"

def run_command(command, output_file=None):
    if output_file:
        with open(output_file, "wb+") as out:
            p = Popen(command, shell=True, stdout=out, stderr=out)
            p.communicate()
    else:
        p = Popen(command, shell=True)
        p.communicate()



def launch_instances(num_to_launch, instance_type, task_tag):
    ec2 = boto3.resource('ec2')
    # Get list of instances that have our
    # exp_id tag and designated instance tpe
    placement_dict = {
        "AvailabilityZone": target_availability_zone,
        "GroupName": target_placement_group
    }
    if instance_type[0] == 'g':
        image_id = gpu_ami
    else:
        image_id = cpu_ami

    launched_instances = ec2.create_instances(
        ImageId=image_id,
        SecurityGroups=security_groups,
        InstanceType=instance_type,
        MinCount=num_to_launch,
        MaxCount=num_to_launch,
        # Placement=placement_dict,
        KeyName=KEYPAIR_NAME,
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": EBS_VOLUME, "VolumeType": "io2", "Iops": 3000}}],
        # InstanceMarketOptions={"MarketType": "spot"},
    )


    instance_ids = [i.id for i in launched_instances]
    response = ec2.create_tags(
        Resources=instance_ids,
        Tags=[
            {
                'Key': 'Name',
                'Value': f"xdl-exp-{INSTANCE_SPECS[2][1]}w"
            },
            {
                'Key': "EXP_ID",
                'Value': f"xdl-exp-{INSTANCE_SPECS[2][1]}w"
            },
            {
                'Key': "TASK_TYPE",
                'Value': task_tag
            }
        ])

    for instance in launched_instances:
        instance.wait_until_running()
        while instance.public_ip_address is None:
            instance.reload()
            time.sleep(2)
    return launched_instances


def init_docker_environment(ip, spec, index):
    processor = spec[2]
    if processor == 'cpu':
        command = "\"sudo rm /var/lib/apt/lists/lock && sudo rm /var/cache/apt/archives/lock && sudo rm /var/lib/dpkg/lock* && sudo dpkg --configure -a " \
                  "&& sudo apt-get update && sudo apt-get install -y docker.io " \
                  "&& sudo docker pull {img}\"".format(img=CPU_DOCKER_IMG)
        run_command(f"ssh -i {KEYPAIR_PATH} -o \"StrictHostKeyChecking no\" ubuntu@" + ip + " " + command,
                    f"{OUTPUT_DIR}/setup_" + ip)
    else:
        run_command(f"ssh -i {KEYPAIR_PATH} -o \"StrictHostKeyChecking no\" ubuntu@" + ip + " \'bash -s\' < gpu.sh", f"{OUTPUT_DIR}/setup_" + ip)

    if spec[0] == "worker":
        df = S3_FILES[index]
        run_command(f"ssh -i {KEYPAIR_PATH} -o \"StrictHostKeyChecking no\" ubuntu@" + ip + " mkdir /xdl_training_samples",
                    f"{OUTPUT_DIR}/setup_download_" + ip)
        command = "\"sudo aws s3 cp s3://criteo-terabytes/{df} {data_path}\"".format(df=df, data_path=SERVER_DATA_PATH)
        run_command(f"ssh -i {KEYPAIR_PATH} -o \"StrictHostKeyChecking no\" ubuntu@" + ip + " " + command,
                    f"{OUTPUT_DIR}/setup_download_" + ip)
    print("Finished initialization for " + ip)


def init_ec2_servers():
    # clean up logs
    os.system(f"rm -rf {OUTPUT_DIR}/setup*")

    # step 1: launch instancess
    instances = []
    for (task_type, num_instances, processor, instance_type) in INSTANCE_SPECS:
        instances += launch_instances(num_instances, instance_type, task_type)
    print("All instances launched!")

    # step 2: save ip addresses
    public_ip_addresses = [inst.public_ip_address for inst in instances]
    private_ip_addresses = [inst.private_ip_address for inst in instances]
    with open(PUBLIC_IP_ADDR_PATH, 'w') as f:
        for ip in public_ip_addresses:
            f.write(str(ip) + '\n')

    with open(PRIVATE_IP_ADDR_PATH, 'w') as f:
        for ip in private_ip_addresses:
            f.write(str(ip) + '\n')

    print("public ip addresses: " + str(public_ip_addresses))
    print("private ip addresses: " + str(private_ip_addresses))

    print("Waiting for {s} secs".format(s=LAUNCH_TO_SSH_SLEEP_DURATION))
    time.sleep(LAUNCH_TO_SSH_SLEEP_DURATION)
    print("Initializing server environment")

    # step 3: init docker environment
    ip_count = 0
    threads = []
    for spec in INSTANCE_SPECS:
        for i in range(spec[1]):
            t = threading.Thread(target=init_docker_environment, args=(public_ip_addresses[ip_count], spec, i))
            print("Initializing ip addr " + public_ip_addresses[ip_count])
            t.start()
            threads.append(t)
            ip_count += 1

    for t in threads:
        t.join()
    print("All instances are initialized!")


def launch_workers_only_experiment(exp_tag):
    with open(PUBLIC_IP_ADDR_PATH) as f:
        line_pub_ips = f.readlines()
    public_ips = [line[:len(line) - 1] for line in line_pub_ips]

    with open(PRIVATE_IP_ADDR_PATH) as f:
        line_pri_ips = f.readlines()
    private_ips = [line[:len(line) - 1] for line in line_pri_ips]
    ip_counter = 0

    scheduler_ip = private_ips[0]
    ssh_command_temp = f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{ip} \"sudo docker start \$(sudo docker ps -aq) && sudo {docker_version} exec " \
                       "\$(sudo docker ps -aq) /bin/bash -c \'apt update && apt install -y default-jdk " \
                       "&& export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre && export HADOOP_HOME=/root/hadoop " \
                       "&& export HADOOP_HDFS_HOME=/root/hadoop " \
                       "&& export HADOOP_CLASSPATH=$(find $HADOOP_HOME -name '*.jar' | xargs echo | tr ' ' ':') " \
                       "&& export CLASSPATH=$CLASSPATH:$HADOOP_CLASSPATH " \
                       "&& export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server:$LD_LIBRARY_PATH " \
                       "&& export PATH=\${{PATH}}:\${{HADOOP_HOME}}/bin:\${{HADOOP_HOME}}/sbin && cd x-deeplearning-redundancy/xdl/build/ " \
                       "&& git remote remove origin && git remote add origin https://github.com/johnzhang1999/x-deeplearning-redundancy.git " \
                       "&& git config --global credential.helper store " \
                       f"&& echo \"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com\" > ~/.git-credentials " \
                       "&& git pull origin {branch} -X theirs && git checkout {branch} " \
                       "&& cmake .. {cmake_options} && make -j$(nproc) && make install_python_lib && cd {experiment_directory} " \
                       "&& {zookeeper_script} python {training_file} --task_name={task} " \
                       "--zk_addr=zfs://{scheduler_ip}:2181/scheduler {run_options}\'\""

    threads = []
    for (task, num_instances, processor, instance_type) in INSTANCE_SPECS:
        for i in range(num_instances):
            ip = public_ips[ip_counter]
            print("Launching for ip addr " + ip)
            ip_counter += 1
            ssh_command_final = ""
            if task == "worker":
                if processor == "cpu":
                    ssh_command_final = ssh_command_temp.format(ip=ip,
                                                                docker_version="docker",
                                                                branch=EXPERIMENT_BRANCH,
                                                                cmake_options="-DTF_BACKEND=1",
                                                                experiment_directory="../examples/criteo",
                                                                zookeeper_script="",
                                                                training_file=TRAINING_FILE,
                                                                task=task,
                                                                scheduler_ip=scheduler_ip,
                                                                run_options="--task_index={task_index} --task_num={task_num}".format(
                                                                    task_index=0, task_num=1))
                elif processor == "gpu":
                    ssh_command_final = ssh_command_temp.format(ip=ip,
                                                                docker_version="nvidia-docker",
                                                                branch=EXPERIMENT_BRANCH,
                                                                cmake_options="-DUSE_GPU=1 -DTF_BACKEND=1 -DCUDA_PATH=/usr/local/cuda-9.0 -DNVCC_C_COMPILER=/usr/bin/gcc-4.8",
                                                                experiment_directory="../examples/criteo",
                                                                zookeeper_script="",
                                                                training_file=TRAINING_FILE,
                                                                task=task,
                                                                scheduler_ip=scheduler_ip,
                                                                run_options="--task_index={task_index} --task_num={task_num}".format(
                                                                    task_index=0, task_num=1))
            print(ssh_command_final)
            t = threading.Thread(target=run_command,
                                 args=(ssh_command_final, f"{OUTPUT_DIR}/exp_" + exp_tag + "_" + task + "_" + str(i)  + ".txt"))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()
    print("Finish launching all experiments")


def launch_experiment(exp_tag):
    with open(PUBLIC_IP_ADDR_PATH) as f:
        line_pub_ips = f.readlines()
    public_ips = [line[:len(line) - 1] for line in line_pub_ips]

    with open(PRIVATE_IP_ADDR_PATH) as f:
        line_pri_ips = f.readlines()
    private_ips = [line[:len(line) - 1] for line in line_pri_ips]
    # clean up servers
    os.system(f"rm -rf {OUTPUT_DIR}/exp_*")
    num_servers = INSTANCE_SPECS[1][1]

    threads = []
    for ip in public_ips:
        t = threading.Thread(target=run_command, args=(f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{ip} \"sudo docker stop \$(sudo docker ps -aq) && sudo docker rm \$(sudo docker ps -aq)\"".format(ip=ip),))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


    ip_counter = 0

    scheduler_ip = private_ips[0]

    ssh_command_temp = f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{ip} \"sudo {docker_version} run " \
                       "-v /home/ubuntu/hadoop:/root/hadoop {data_path} " \
                       "--network=host {bg} {image} /bin/bash -c \'apt update && apt install -y default-jdk " \
                       "&& export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre && export HADOOP_HOME=/root/hadoop " \
                       "&& export HADOOP_HDFS_HOME=/root/hadoop " \
                       "&& export HADOOP_CLASSPATH=$(find $HADOOP_HOME -name '*.jar' | xargs echo | tr ' ' ':') " \
                       "&& export CLASSPATH=$CLASSPATH:$HADOOP_CLASSPATH " \
                       "&& export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server:$LD_LIBRARY_PATH " \
                       "&& export PATH=\${{PATH}}:\${{HADOOP_HOME}}/bin:\${{HADOOP_HOME}}/sbin && cd x-deeplearning-redundancy/xdl/build/ " \
                       "&& git remote remove origin && git remote add origin https://github.com/johnzhang1999/x-deeplearning-redundancy.git " \
                       "&& git config --global credential.helper store " \
                       f"&& echo \"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com\" > ~/.git-credentials " \
                       "&& git pull origin {branch} -X theirs && git checkout {branch} " \
                       "&& cmake .. {cmake_options} && make -j$(nproc) && make install_python_lib && cd {experiment_directory} " \
                       "&& {zookeeper_script} python {training_file} --task_name={task} " \
                       "--zk_addr=zfs://{scheduler_ip}:2181/scheduler {run_options}\'\""
    hdfs_ip = ""
    try:
        with open(HDFS_PRIVATE_IP_ADDR_PATH) as hdfsf:
            hdfs_ip = hdfsf.readline()[:-1]
    except Exception:
        pass

    threads = []
    for (task, num_instances, processor, instance_type) in INSTANCE_SPECS:
        for i in range(num_instances):
            ip = public_ips[ip_counter]
            print("Launching for ip addr " + ip)
            ip_counter += 1
            ssh_command_final = []
            if task == "scheduler":
                ssh_command_final.append(ssh_command_temp.format(data_path="",
                                                            ip=ip,
                                                            docker_version="docker",
                                                            bg="",
                                                            image=CPU_DOCKER_IMG,
                                                            branch=EXPERIMENT_BRANCH,
                                                            cmake_options="-DTF_BACKEND=1",
                                                            experiment_directory="../examples/criteo",
                                                            zookeeper_script=ZOOKEEER_SETUP_SCRIPT,
                                                            training_file=TRAINING_FILE,
                                                            task=task,
                                                            scheduler_ip="localhost",
                                                            run_options="--ps_num={num_servers} --ps_cpu_cores={ps_num_cores} "
                                                                        "--ps_memory_m={ps_memory_mb} --ckpt_dir={ckpt_dir}".format(
                                                                num_servers=num_servers, ps_num_cores=PS_NUM_CORES,
                                                                ps_memory_mb=PS_MEMORY_MB, ckpt_dir=CKPT_DIR.format(ip=hdfs_ip))))
            elif task == "ps":
                ssh_command_final.append(ssh_command_temp.format(data_path="",
                                                            ip=ip,
                                                            docker_version="docker",
                                                            bg="",
                                                            image=CPU_DOCKER_IMG,
                                                            branch=EXPERIMENT_BRANCH,
                                                            cmake_options="-DTF_BACKEND=1",
                                                            experiment_directory="../examples/criteo",
                                                            zookeeper_script="",
                                                            training_file=TRAINING_FILE,
                                                            task=task,
                                                            scheduler_ip=scheduler_ip,
                                                            run_options="--task_index={task_index}".format(
                                                                task_index=i)))
            elif task == "worker":
                if processor == "cpu":
                    for task_ind in range(NUM_WORKER_TASKS_PER_INSTANCE):
                        ssh_command_final.append(ssh_command_temp.format(data_path="-v {dp}:{dp}".format(dp=SERVER_DATA_PATH),
                                                                    ip=ip,
                                                                    docker_version="docker",
                                                                    bg="",
                                                                    image=CPU_DOCKER_IMG,
                                                                    branch=EXPERIMENT_BRANCH,
                                                                    cmake_options="-DTF_BACKEND=1",
                                                                    experiment_directory="../examples/criteo",
                                                                    zookeeper_script="",
                                                                    training_file=TRAINING_FILE,
                                                                    task=task,
                                                                    scheduler_ip=scheduler_ip,
                                                                    run_options="--task_index={task_index} --task_num={task_num}".format(
                                                                        task_index=(i * NUM_WORKER_TASKS_PER_INSTANCE + task_ind) % 15,
                                                                        task_num=num_instances * NUM_WORKER_TASKS_PER_INSTANCE
                                                                    )))
                elif processor == "gpu":
                    for task_ind in range(NUM_WORKER_TASKS_PER_INSTANCE):
                        ssh_command_final.append(ssh_command_temp.format(data_path="-v {dp}:{dp}".format(dp=SERVER_DATA_PATH),
                                                                    ip=ip,
                                                                    docker_version="nvidia-docker",
                                                                    bg="-e NVIDIA_VISIBLE_DEVICES={gpu_ind}".format(gpu_ind=task_ind),
                                                                    image=GPU_DOCKER_IMAGE,
                                                                    branch=EXPERIMENT_BRANCH,
                                                                    cmake_options="-DUSE_GPU=1 -DTF_BACKEND=1 -DCUDA_PATH=/usr/local/cuda-9.0 -DNVCC_C_COMPILER=/usr/bin/gcc-4.8",
                                                                    experiment_directory="../examples/criteo",
                                                                    zookeeper_script="",
                                                                    training_file=TRAINING_FILE,
                                                                    task=task,
                                                                    scheduler_ip=scheduler_ip,
                                                                    run_options="--task_index={task_index} --task_num={task_num}".format(
                                                                        task_index=i * NUM_WORKER_TASKS_PER_INSTANCE + task_ind,
                                                                        task_num=num_instances * NUM_WORKER_TASKS_PER_INSTANCE
                                                                    )))
            print(ssh_command_final)

            for ssh_cmd_ind in range(len(ssh_command_final)):
                ssh_cmd = ssh_command_final[ssh_cmd_ind]
                t = threading.Thread(target=run_command,
                                     args=(ssh_cmd, f"{OUTPUT_DIR}/exp_" + exp_tag + "_" + task + "_" + str(i) + "_"+ str(ssh_cmd_ind) + ".txt"))
                t.start()
                threads.append(t)
                if task == "scheduler":
                    time.sleep(SCHEDULER_DELAY_DURATION)

    for t in threads:
        t.join()
    print("Finish launching all experiments")


def setup_hdfs():
    # step 1: launch instancess
    instances = launch_instances(HDFS_CLUSTER_SIZE, HDFS_CLUSTER_INSTANCE_TYPE, "hdfs_server")
    print("All instances launched!")

    # step 2: save ip addresses
    public_ip_addresses = [inst.public_ip_address for inst in instances]
    private_ip_addresses = [inst.private_ip_address for inst in instances]
    with open(HDFS_PUBLIC_IP_ADDR_PATH, 'w') as f:
        for ip in public_ip_addresses:
            f.write(str(ip) + '\n')

    with open(HDFS_PRIVATE_IP_ADDR_PATH, 'w') as f:
        for ip in private_ip_addresses:
            f.write(str(ip) + '\n')

    with open ("hdfs_setup/workers", "w") as f:
        for j in range(1, len(private_ip_addresses)):
            f.write(str(private_ip_addresses[j]) + "\n")

    with open("hdfs_setup/core-site.xml", "w") as f:
        f.write(CORE_SITE_TEMPLATE.format(ip=private_ip_addresses[0]))

    with open("hdfs_setup/yarn-site.xml", "w") as f:
        f.write(YARN_SITE_TEMPLATE.format(ip=private_ip_addresses[0]))

    time.sleep(LAUNCH_TO_SSH_SLEEP_DURATION)
    print("Initializing hdfs environment")
    command =  f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{pub_ip} " \
               "\"cd && rm -rf hadoop* && wget http://apache.mirrors.hoobly.com/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz " \
               "&& tar -xzf hadoop-3.2.1.tar.gz && mv hadoop-3.2.1 hadoop && sudo apt update " \
               "&& sudo apt install -y default-jdk " \
               "&& echo \\\"PATH=/home/ubuntu/hadoop/bin:/home/ubuntu/hadoop/sbin:$PATH\\\" >> /home/ubuntu/hadoop/.profile " \
               "&& echo 'export HADOOP_HOME=/home/ubuntu/hadoop' >> ~/.bashrc " \
               "&& export HADOOP_HOME=/home/ubuntu/hadoop " \
               "&& echo 'export PATH=\${{PATH}}:\${{HADOOP_HOME}}/bin:\${{HADOOP_HOME}}/sbin' >> ~/.bashrc " \
               "&& export PATH=\${{PATH}}:\${{HADOOP_HOME}}/bin:\${{HADOOP_HOME}}/sbin " \
               "&& echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre' >> ~/.bashrc " \
               "&& export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre " \
               "&& source ~/.bashrc \"" \
               f"&& scp -i {KEYPAIR_PATH} " + "hdfs_setup/* ubuntu@{pub_ip}:hadoop/etc/hadoop/" \
               f"&& ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{pub_ip} \" source ~/.bashrc\""


    threads = []

    with open(PUBLIC_IP_ADDR_PATH) as f:
        line_pub_ips = f.readlines()
    non_hdfs_public_ips = [line[:len(line) - 1] for line in line_pub_ips]

    all_public_ips = public_ip_addresses + non_hdfs_public_ips

    for pub_ip in all_public_ips:
        c = command.format(pub_ip=pub_ip)
        print(c)
        t = threading.Thread(target=run_command, args=(c,  f"{OUTPUT_DIR}/hdfs_setup_"+pub_ip))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("Mounting disks")
    command = f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@{pub_ip} \"sudo file -s /dev/nvme1n1 " \
              "&& sudo mkfs -t xfs /dev/nvme1n1 && sudo mkdir /mydata && sudo mount /dev/nvme1n1 /mydata " \
              "&& sudo chown -R ubuntu:root /mydata \""
    for ip in public_ip_addresses:
        os.system(command.format(pub_ip=ip))
    print("Done with hdfs setup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", type=str, choices=["init", "init_launch_only", "run", "run_workers_only", "summarize", "setup_hdfs", "send_instruction"],
                        help="Mode of the script. Either run or init.")
    parser.add_argument('branch', type=str, default=EXPERIMENT_BRANCH)
    parser.add_argument('num_workers', type=int, default=15, help='Number of workers to launch')
    args = parser.parse_args()

    def get_timestamp():
        return str(time.time()).split('.')[0]

    OUTPUT_DIR = f'out_num_workers_24xlarge_0420/{args.branch}/{args.num_workers}'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if args.run_mode == "init":
        assert INSTANCE_SPECS[2][0] == 'worker'
        INSTANCE_SPECS[2] = ('worker', args.num_workers, 'gpu', 'p3.2xlarge')
        print("########## Initializing instances with specs: ##########")
        print(INSTANCE_SPECS)
        # 1/0
        init_ec2_servers()
    elif args.run_mode == "run":
        assert INSTANCE_SPECS[2][0] == 'worker'
        INSTANCE_SPECS[2] = ('worker', args.num_workers, 'gpu', 'p3.2xlarge')
        print("########## Running experiments with specs: ##########")
        print(INSTANCE_SPECS)

        EXPERIMENT_BRANCH = args.branch
        print('>>> Experiment branch:', EXPERIMENT_BRANCH)

        # 1/0
        launch_experiment("init_run")
    elif args.run_mode == "run_workers_only":
        print("Running experiments(workers only) with specs:")
        print(INSTANCE_SPECS)
        for i in range(10):
            print("Round repeat " + str(i))
            launch_workers_only_experiment(str(i))
    elif args.run_mode == "summarize":
        for i in range(NUM_REPEAT):
            maxx = 0.0
            for filename in os.listdir(OUTPUT_DIR):
                if filename.startswith("exp_"+str(i)):
                    with open(f"{OUTPUT_DIR}/" + filename) as f:
                        lines = f.readlines()
                        val = 0
                        try:
                            val = eval(lines[-3])
                        except Exception:
                            print(lines[-3])
                            try:
                                eval(lines[-2])
                            except Exception:
                                print(lines[-2])
                                try:
                                    eval(lines[-1])
                                except Exception:
                                    print(lines[-1])
                        if val == 0:
                            print("gg for file " + filename)
                        else:
                            maxx = max(maxx, val)
            print("Exp {i}: {maxx}".format(i=i, maxx=maxx))
    elif args.run_mode == "setup_hdfs":
        setup_hdfs()
    elif args.run_mode == "send_instruction":
        with open(HDFS_PUBLIC_IP_ADDR_PATH) as f:
            line_pub_ips = f.readlines()
        with open(PUBLIC_IP_ADDR_PATH) as f:
            line_pub_ips += f.readlines()
        public_ips = [line[:len(line) - 1] for line in line_pub_ips]
        for ip in public_ips:
            print("Sending to " + ip)
            os.system(f"ssh -i {KEYPAIR_PATH} " + "-o \"StrictHostKeyChecking no\" ubuntu@" + ip + " \"{cmd}\"".format(cmd=INSTRUCTION_TO_SEND))
