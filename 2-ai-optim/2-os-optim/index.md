---
author: "Hongyao Tang"
title: "2. OS Optimization"
date: "2025-07-02"
description: "Optimization for Linux OS"
tags: [
    "optimization",
]
ShowToc: true
weight: 2
---


## Orchestrator

#### Job scheduler
- SLURM is deployed for training clusters 
- Kubernetes is typically favored for inference clusters
- Open-source Slinky project and CoreWeave’s SUNK product are examples of these integrated solutions to simplify cluster management across training and inference workloads
  
#### S - HW-topology-unaware scheduling
- Pod placements. The requirement is allocating resources to containers in a manner that is aware of the hardware topology including the NUMA node and network bandwidth configurations
- Kubernetes is not topology aware. It treats each GPU as a resource but doesn’t know if GPU0 and GPU1 are on the same NUMA node or if they use the same NVLink interconnect.

T - Use node labels to explicitly mark GPU or node topology.

- Ensures that multi-GPU pods are placed on hosts where the GPUs share the same NVLink domain or NUMA node.

T - Kubernetes Topology Manager and Nvidia’s hardware-aware GPU Operator **device plugin** can provide detailed topology information. 

- they can detect that GPU 0 is connected to NUMA node 0, NVLink domain A, and PCIe bus Z.
- Kubernetes scheduler can then use this information to allocate containers to GPUs



A

- simply request a GPU in your Kubernetes pod specification
- device plugin mounts the /dev/nvidia* devices for you
- device plugin also mounts the driver libraries from the host into the container.


R

ensure optimal performance by minimizing cross-NUMA-node communication and reducing latency in multi-node GPU workloads.


#### S - Resource contention for performance-sensitive Kubernetes jobs

| 资源类型 | 是否可压缩 | 说明 |
|----------|-------------|------|
| CPU | ✅ 可压缩 | CPU 使用可以被限制（throttle），Pod 会变慢但不会被杀死。 |
| Memory | ❌ 不可压缩 | 内存使用超出限制会导致 OOM（Out of Memory）并被杀死。 |
- a greedy container, if unconstrained, could allocate too much memory on the host. This would cause the host to swap some of its memory to disk.
- If an unbounded container uses too much memory on the host, the infamous Linux “OOM Killer” will start killing processes - and potentially your Kubernetes job - even if your job wasn’t the one using too much memory.
- The OOM killer uses heuristics when deciding which pods to kill. Sometimes it decides to kill the largest running pod which is likely your large training or inference job holding lots of data in CPU RAM to feed the GPUs.



T - Guaranteed QoS

Kubernetes根据**Pod中容器**的资源请求和限制（`requests` 和 `limits`）将其分为三种 QoS 类别：
| QoS 类别 | 条件 | 特点 |
|----------|------|------|
| **Guaranteed** | 所有容器的 `requests` 和 `limits` 都设置了，并且值相等 | 最稳定，优先级最高，最不容易被驱逐 |
| **Burstable** | 至少一个容器设置了 `requests`，但 `requests` 和 `limits` 不完全相等 | 中等优先级，资源使用灵活 |
| **BestEffort** | 所有容器都没有设置 `requests` 和 `limits` | 优先级最低，最容易被驱逐 |


T - Exclusively, request all of the CPUs and GPUs of a given node so that nothing else interferes or contends with the jobs’ resources.


## Container
### Image
#### S -  “but it works on my machine” problem
T - dependency as readonly layer

- include CUDA lib
- making sure that the CUDA libraries inside the container match the driver on the host.


R

dependencies versions are consistent


#### S - Large image size slow container startup
- Container startup times can be quite a bit slower if the image is huge and needs to be pulled over the network. 

T

- Maultistage build: remove build tool and tem build file
- Multi-layer build: to share layers with ohter images

R

saves disk space and improves container startup time.



### Overlay to host
#### Why use container not VM?
In contrast to VM’s, containers share the host OS kernel, has near bare metal performance
- CPU, memory, FS, network operations perform at near-native speed
- directly use the host’s GPU driver and hardware
  - 容器 → Host OS → 驱动 → 硬件
  - Guest OS → 虚拟化层（hypervisor）→ Host OS → 驱动 → 硬件



#### S - [FS] Overlay(to host) filesystem has overhead when IO with disk
- The main difference when running in a Docker container versus running directly on the host might be in I/O.
- There is some overhead when using an overlay filesystem
  - This extra latency arises because the filesystem must check multiple underlying layers - both read-only and writable - to determine which version of a file should be returned.
  - Furthermore, when writing, the copy-on-write (CoW) mechanism used by the overlay. 
- But model training often involves heavy I/O operations when reading datasets, loading a model, and writing model checkpoints re. the FS.

t - Mount a host/network directory into the container using bind mounts.
- inpout data/model directory
- output checkppint directory


A

-v /data/dataset:/mnt/dataset:ro

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-demo
spec:
  containers:
  - name: busybox
    image: busybox
    command: [ "sleep", "3600" ]
    volumeMounts:
    - name: host-volume
      mountPath: /data/host
  volumes:
  - name: host-volume
    hostPath:
      path: /data/on/host
      type: Directory
```
- 宿主机目录（hostPath）中已有数据
- 容器中挂载点（mountPath）也已有数据（比如镜像中预置的文件）
- 容器启动后，宿主机目录会覆盖容器内原有目录
- 容器中原本在该路径下的文件将不可见，但不会被删除（只是被挂载层隐藏）

R

ensure that I/O is not bottlenecked by the overhead of the container’s copy-on-write mechanism.



#### S - [Net] Overlay(to host) network/NAT has overhead when communicating internally and externally
```
Pod ←→ Pod：通过 Overlay 网络通信（可能封装，但不 NAT）
Pod → 外部：通过 SNAT（源地址转换）
外部 → Pod：通过 DNAT（目标地址转换）
```
- When you run multi-node GPU workloads using containers with Kubernetes, the pods need to talk to each other.
- In Kubernetes, by default pods have their own IP and there might be an overlay network or network-address translation (NAT) between pods on different nodes. 
- This can introduce complications and additional overhead.

T - Use hots network

- container’s network is not isolated as it uses the host’s network interface directly.

A

--network=host

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostnetwork-demo
spec:
  hostNetwork: true
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```


R

Using host networking allows a container to access the InfiniBand interconnect exactly as the host does - without any additional translation or firewall layers


## OS - Pin to dedication
OS的设计原本是为了多任务的交替，ML需要专一提高性能，所有要Pin住某些交替机制，保持稳定，减少overhead

### CPU for GPU
#### Why don't GPUs just replace CPUs? 
- Each has its talent
- Formula-1 race cars and Dump trucks both exist at the same time because they're designed for different jobs.
- GPUs are extremely good at doing a very narrow set of instructions
  - namely the type of instructions needed to calculate the pixels of an image.
- Wheras a CPU is kinda good at doing a vast, vast range of possible instructions/calculations, including those that GPUs would really struggle with.
  - complex instruction sets, data processing in RAM, I/O operations, and operating system management

#### S - [绑定thread到专用CPU]Thread is interrupted by os scheduler
- a very latency-sensitive thread that feeds the GPU with data
- Linux by default uses the Completely Fair Scheduler (CFS), impacting the thread

T - Use real-time first-in-first-out (FIFO) or round-robin (RR) scheduling for that thread


实时（Real-Time）在计算机系统中，系统不仅要正确地完成任务，还要在规定的时间内完成。再规定的时间内完成，不是无限制，感觉像是实时完成的
- 硬实时：飞机起飞时，传感器必须在 10 毫秒内响应，否则可能导致事故。
- 软实时：视频播放器如果偶尔卡顿 0.1 秒，用户可能感觉不舒服，但不会出错。
- 非实时：你打开一个网页，等 3 秒也没关系，只是体验差一点。


对比
| 特性 | CFS（Completely Fair Scheduler） | FIFO（First-In-First-Out） | RR（Round-Robin） |
|------|----------------------------------|-----------------------------|--------------------|
| **调度类型** | 非实时调度器 | 实时调度器 | 实时调度器 |
| **调度策略** | 基于虚拟运行时间（vruntime）公平调度 | 按优先级 + 到达顺序执行 | 按优先级 + 时间片轮转 |
| **优先级机制** | nice 值（-20 ~ +19） | 实时优先级（1~99） | 实时优先级（1~99） |
| **时间片支持** | ✅ 动态分配 | ❌ 无时间片 | ✅ 固定时间片（如 100ms） |
| **抢占机制** | ✅ **支持（低优先级被抢占）**| ✅ **仅被更高优先级抢占**| ✅ **同 FIFO，但同优先级轮转** |
| **公平性** | ✅ 高（基于运行时间） | ❌ 无（先来先服务） | ✅ 同优先级公平轮转 |
| **饥饿风险** | ✅ 有（低 nice 值进程可能饿死） | ✅ 有（低优先级可能永远不运行） | ✅ 有（低优先级可能饿死） |
| **适用场景** | 普通用户进程、后台任务 | 硬实时任务、驱动控制 | 多个实时任务共享 CPU |
| **内核接口** | `SCHED_NORMAL` / `SCHED_OTHER` | `SCHED_FIFO` | `SCHED_RR` |
| **典型应用** | 浏览器、数据库、Web 服务 | 飞控系统、工业控制 | 实时音频、机器人控制 |


A

1.Pod 配置中启用 hostPID 和 hostNetwork（可选）
- hostPID: true 允许容器看到宿主机的进程 ID，便于调试和设置调度策略
```yaml
spec:
  hostPID: true
  hostNetwork: true
  containers:
  - name: ml-worker
    image: your-ml-image
    securityContext:
      xxx
```

2.授予 CAP_SYS_NICE 权限
- 允许容器内的进程使用 sched_setscheduler() 设置为 SCHED_FIFO
```yaml
securityContext:
  capabilities:
    add: ["SYS_NICE"]
```

3.在容器中设置调度策略
- 可以在容器启动脚本或程序中使用 chrt 命令或 C 语言 API 设置调度策略
  - -f 表示 FIFO
  - 80 是实时优先级（1~99）
```bash
chrt -f 80 ./your_ml_program
```

4.确保宿主机允许实时调度
```bash
# systemd
[Service]
LimitRTPRIO=99
```

5.summary
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fifo-ml-pod
spec:
  hostPID: true
  hostNetwork: true
  containers:
  - name: ml
    image: your-ml-image
    securityContext:
      capabilities:
        add: ["SYS_NICE"]
    command: ["chrt", "-f", "80", "./your_ml_program"]
```


R

- ensure the runs without being preempted by normal threads

缺点
- However, use this with caution as real-time threads can starve other processes if not managed properly. 
- In practice, however, if you’ve pinned your threads to dedicated cores, you often don’t need to mess with real-time thread priorities, but it’s worth keeping an eye on.


T2 - isolate cores entirely for your process from scheduler

A2

1.Defining isolated CPUs
```bash
# /sys/devices/system/cpu/isolated
isolcpus=<cpu number>,....,<cpu number>
```

2.assigning processes/tasks to or from the CPU
```bash
# taskset command can be used to set the CPU affinity of processe
taskset -c <CPU NUMBER> -p <PID>

# cset command allows you to group CPUs and memory into logical entities and restrict processes to the resources of one or more of those logical entities.
cset set --cpu <CPU> <CPUSET NAME> # 创建 CPU 集合
cset proc --set=rt --exec ./your_program # 启动进程并绑定
cset proc --move --pid=<PID>,...,<PID> --toset=<CPUSET NAME> # 绑定已有进程
taskset -cp <PID> # 验证绑定效果
```

R2

OS scheduler leaves those CPU cores for you to use as your program wishes.



#### S - [pin cpu,绑定thread到特定CPU]Bind thread to a NUMA node of CPU and memory
- CPU-based worker processes - CPU is responsible for preparing the next batch of data including loading the data from disk, tokenizing the data, transforming it, etc.
- by default, the Linux scheduler will not use a NUMA-aware scheduling algorithm.

T - NUMA node

- A NUMA node is a logical grouping of CPUs, GPUs, NICs, and memory that are physically close to each other.
- Accessing resources within a single NUMA node is faster than accessing resources in other NUMA nodes.

A

```bash
# 1.查看详细信息
numactl -H
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 ... 15
# node 0 size: 128000 MB
# node 1 cpus: 16 17 ... 31
# node 1 size: 128000 MB

# 2.NUMA 节点是由 主板和 CPU 架构决定的，不能通过软件创建新的 NUMA 节点。你只能：


# 3.将进程/线程分配到特定 NUMA 节点
numactl --cpubind=0 --membind=0 python train.py
# --cpubind=<node>	# 将进程绑定到指定 NUMA 节点的 CPU 上
# --membind=<node>	# 将进程使用的内存限制在指定 NUMA 节点

# 4.查看内存访问
numastat -p $(pidof python)
```

```py
import numa
numa.available_nodes()  # 查看可用节点
numa.bind(0)            # 绑定到节点 0
```

#### S - [绑定硬件中断到特定CPU]Bind hardware interrupts to a NUMA node of CPU
- If your GPU or NIC running in a NUMA node 0 generates hardware interrupts, you’d like those to be handled by a core on the same NUMA node. 
- Otherwise, a random core from another NUMA node has to communicate across NUMA nodes.
- This could evict useful cache data on the other NUMA node.

T - bind hardware interrupts to specific cores in a NUMA-aware manner


A

irqbalance 会读取系统的 NUMA 拓扑（通过 /sys 和 ACPI 表），并尝试将设备的中断分配到与设备在同一 NUMA 节点的 CPU 上。
```bash
systemctl start irqbalance
```


cat /proc/interrupts

R

Avoid cross-NUMA-node interrupts that could evict useful cache data on the other NUMA node.


#### S - CPU dwonclock or sleep

| 分类项   | ⚡ P-state（Performance State性能状态）                                       | 😴 C-state（Idle State空闲状态）                                             |
|----------|--------------------------------------------------------------|---------------------------------------------------------------------|
| 定义     | 运行时通过调整频率和电压来节能                               | CPU 核心空闲时进入不同睡眠级别                                     |
| 特点     | - P0 是最高性能状态<br>- P1、P2… 是逐渐降低频率和电压的状态<br>- 由操作系统或硬件动态调整（DVFS） | - C0：活跃状态<br>- C1：轻度睡眠，快速唤醒<br>- C6：深度睡眠，几乎断电，唤醒慢 |
| 关键词   | downclock（降频）                                            | sleep（睡眠）                                                      |
- many compute nodes will run CPUs in a power-saving mode which 
  - either downclocks a CPU 
  - or puts it to sleep when it’s idle
- This helps save energy, reduce heat, and lower cost. 
- tThese power management features could cause extra latency when the system wakes the CPUs up again when new work arrives.
  ○ Bubbles are periods of time when the GPU is waiting for the CPU to resume data processing.


T -  trade a bit of extra power draw for more responsive CPU behavior.

- configure the CPU frequency governor to “performance” mode which keeps the CPU at max frequency all the time. 
- disabling deep C-states can keep cores from going into a low-power sleep state

A
```bash
cpupower frequency-set -g performance


# /etc/default/grub
# 在 GRUB_CMDLINE_LINUX_DEFAULT 中添加：
intel_idle.max_cstate=0 processor.max_cstate=1
update-grub
reboot
```


R

- Avoid bubbles/reduce hiccups
- To keep GPU fed all the potential time
- For maximum and consistent performance



### Memory for GPU
#### S - [Page的单位]Large data may have many pages if using small page size
- big-memory workloads - when you have processes using tens or hundreds of gigabytes of memory
- Linux memory management typically uses 4 KB pages
- managing millions of tiny pages is inefficient


T - Transparent Huge Pages (THP) 2 MB (default) or even 1 GB pages

A

```bash
# 启用系统范围内的 THP
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
# - always：系统会尽可能使用 THP
# - madvise：仅在应用程序使用 madvise() 标记的内存区域启用 THP（推荐）
# - never：完全禁用 THP
```

R

Reduce the overhead of virtual memory management
- fewer page faults, a few percent improvement in throughput
- less pressure on the Translation Lookaside Buffer (TLB).
  - The TLB is a cache that the CPU uses to map virtual addresses to physical ones. 
  - Fewer, larger pages means the TLB can cover more memory with the same number of entries


#### S - [pin mem, Page的交换]Page swapping impact data transferring to GPU
- When transferring/copy data from CPU to the GPU
- Normally, the OS can decide to swap memory pages in and out - or move them around as needed.

T - page pinning/page locking/memory pinning to ensure pages in RAM not disk

- if you allocate pinned memory, the OS guarantees those memory pages will stay in physical RAM and not be swapped out or moved

A

1.The OS has a limit on how much memory a user can lock (pin). Typically, one sets it to unlimited for large AI workloads and HPC applications.
```bash
ulimit -l <max locked memory in KB>
ulimit -l 65536  # 设置为 64MB
```

2.tells Linux to avoid swapping except under extreme memory pressure.
```bash
/etc/sysctl.conf
vm.swappiness = 0

sysctl -p
```
- PyTorch’s DataLoader has a flag pin_memory=True which, when true, means the batches loaded will be placed in pinned RAM.
```py
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义数据集
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 创建 DataLoader，启用 pin_memory
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
```


R

pinned (page-locked) host memory can dramatically improve throughput.
- Copying from pinned host CPU memory to a GPU is often 2–3× faster than from regular pageable CPU memory.


### FS
#### S - OS FS page cache/flush too small/freuqent for large data write-back
- Write frequent checkpoints to disk in case you need to restart a failed job from a known good checkpoint
- During checkpointing, however, huge bursts of data might fill up the OS page cache and cause stalls.

T -  allow a lot of dirty cache to accumulate and flush in the background

A

vm.dirty_ratio - 当脏页占用内存达到该比例时，写入数据的进程会被阻塞，直到脏页被写入磁盘
vm.dirty_background_ratio - 当系统中脏页占用内存达到该比例时，后台线程开始异步写入磁盘
```bash
 /etc/sysctl.conf
vm.dirty_background_ratio = 10
vm.dirty_ratio = 20

sysctl -p
```

R

Training process doesn’t block on file writes


### GPU itself (Driver)
#### S - How OS to interface with GPUs HW
- manages low-level GPU operations including memory allocation on the device, task scheduling on GPU cores, and partitioning the GPU for multi-tenant usage.

T - driver

NVIDIA driver will
- install kernel modules
- create device files like /dev/nvidia0

Tools like nvidia-smi come with the driver 
- allow you to monitor temperatures, measure utilization, query error-correcting code (ECC) memory status, and enable different GPU modes like persistence mode.






- Normally, when multiple processes share a single GPU, the GPU’s scheduler time-slices between them. 
- If those kernels are short and there’s an idle gap between them, the GPU can end up underutilized as it’s 
  - doing “ping-pong” context switches and
  - not overlapping the work.


T1 - Umbrella with NVIDIA’s Multi-Process Service (MPS) 
- A feature that creates a sort of umbrella under which multiple processes can run on the GPU concurrently and without strict time slicing.
  - merges the contexts of the processes into one scheduler context
- With MPS, the GPU can execute kernels from different processes at the same time as long as the GPU resources (streaming multiprocessors, tensor cores, etc.) are available
  - overlapping
  - an active thread percentage per client. This limits how many streaming multiprocessors (GPU cores, essentially) a client can use. This can be useful if you want to guarantee quality of service (QoS) where two jobs, for example, each get at most 50% of the GPU’s execution resources. If not explicitly set, the jobs will just compete and use whatever GPU resources they can.


A1

1.running an MPS control daemon 
- nvidia-cuda-mps-control
- which then launches an MPS server process that brokers GPU access
- start the MPS server on a node - often one per GPU or one per user
2.run your GPU jobs with an environment variable that connects them to MPS
3.All jobs under that server will share the GPU concurrently


R1

don’t pay the full cost of switching and idling between independent processes.
 
缺点

- Note that MPS does not partition GPU memory, so all processes will share the full GPU memory space. MPS is mainly about compute sharing and scheduling. The issue is that one process could request a massive amount of GPU RAM, cause an out-of-memory (OOM) error on the GPU, and result in terminating all of the other processes running on the GPU. This is very disruptive. 
- Another limitation of MPS is that, by default, all MPS clients must run as the same Unix user since they share a context.




T2 - partition GPU into a multi-instance GPU(MIG)

- Starting with the NVIDIA A100 Ampere generation, GPUs can be partitioned at the hardware level into multiple instances using multi-instance GPU.
- MIG allows a GPU to be sliced into as many as 7 smaller logical GPUs - each with its own dedicated portion of memory and compute units, or streaming multiprocessors (SMs). 
  - A 192 GB Blackwell B200 GPU can be split into 7 instances of about 27 GB (192 GB / 7 instances) and 20 SMs (140 SM’s / 7 instances) each
- Each instance acts like a separate GPU from the perspective of software since it has its own memory, its own streaming multiprocessors, and even separate engine contexts.

A

- The GPU has to be put into MIG mode
- the slices created, 
- the node rebooted, and then the slices appear as separate devices to the system.
The Kubernetes device plugin will list MIG devices as resources like “nvidia.com/mig-2g.10gb” in the case of a 2 GPU slice of 10 GB.


```bash
# 启用 GPU 0 的 MIG 模式
 nvidia-smi -i 0 -mig 1

# 查看可用的分区配置（Profile）
nvidia-smi mig -lgip
# Profile ID | GI Size | CI Count | Memory Size
# 9          | 3g      | 3        | 20GB
# 19         | 1g      | 1        | 5GB


# 创建 GPU 实例（GI）
 nvidia-smi mig -cgi 9 -C -i 0
# -cgi 9 使用 Profile ID 为 9 的配置（如 3g.20gb）
# -C     执行创建操作（Create）
# -i 0   指定 GPU 编号为 0


# 查看已创建的实例
nvidia-smi mig -lgi
# 查看 MIG 实例的 UUID（用于容器绑定）
nvidia-smi -L
# GPU 0: A100-SXM4-40GB (UUID: GPU-xxxx)
#   MIG 3g.20gb Instance (UUID: MIG-xxxx)


# 在 Docker 中使用 MIG 实例
docker run -d \
  --gpus "device=MIG-UUID" \
  nvidia/cuda:11.0.3-base-ubuntu20.04 \
  tail -f /dev/null



# Kubernetes 中使用 MIG（简要）
# 安装device plugin支持额外的resources
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo add nvgfd https://nvidia.github.io/gpu-feature-discovery
helm repo update
export MIG_STRATEGY=single  # 或 mixed
helm install nvdp nvdp/nvidia-device-plugin \
  --version=0.7.0 \
  --set migStrategy=${MIG_STRATEGY}
helm install nvgfd nvgfd/gpu-feature-discovery \
  --version=0.2.0 \
  --set migStrategy=${MIG_STRATEGY}


apiVersion: v1
kind: Pod
metadata:
  name: mig-test
spec:
  containers:
  - name: gpu-test
    image: nvcr.io/nvidia/pytorch:23.07-py3
    command: ["nvidia-smi"]
    args: ["-L"]
    resources:
      limits:
        nvidia.com/mig-1g.5gb: 1
```


R2

- Seprate compute
- Seperate memory
- Seperate user


缺点

- If one instance is idle, it can’t lend its resources to another as they are hard partitioned. 
- Also, if you’re not using all MIG slices, you’re wasting resources by leaving them fragmented. It’s important to plan partition sizes to match your workloads.


#### S - dynamic MIG toggle
- run MIG during the day when lots of small training or inferencing experiments are happening
- turn MIG off at night to run big training jobs that use whole GPUs.
- dynamically



T1 - nvidia-mig-parted + crontab
1.开关本身：nvidia-mig-parted
- 这是 NVIDIA 官方的 GPU 分区管理工具，可 declaratively 定义 MIG 模板并应用
- 启用 MIG 并划分为 7 个 1g.5gb 实例
2.自动切换：使用 cron 在白天和晚上自动切换

A1

```bash
version: v1
mig-configs:
  mig-on:
    - devices: all
      mig-enabled: true
      mig-devices:
        "1g.5gb": 7
  mig-off:
    - devices: all
      mig-enabled: false

nvidia-mig-parted apply -f config.yaml -c all-1g.5gb
```

```bash
# 白天启用 MIG（每天早上 8 点）
0 8 * * * /usr/local/bin/nvidia-mig-parted apply -f /etc/mig/config.yaml -c mig-on

# 晚上关闭 MIG（每天晚上 8 点）
0 20 * * * /usr/local/bin/nvidia-mig-parted apply -f /etc/mig/config.yaml -c mig-off
```



T2 - HAMi 
- 是一个 Kubernetes GPU 虚拟化调度插件，支持 自动根据任务类型动态调整 MIG 模板

A2

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hami-device-plugin
  namespace: kube-system
data:
  nodeconfig: |
    - name: MIG-NODE-A
      operatingmode: mig
      scheduleWindow:
        day: "08:00-20:00"
        night: "20:00-08:00"
      migTemplates:
        day: "1g.5gb x 7"
        night: "disable"
```

R2

- 不需要手动执行 nvidia-smi 或 mig-parted
- 根据任务资源请求自动选择合适的 MIG 模板
- 支持统一资源池（MIG + 非 MIG 节点混合调度）
- 推荐






#### S - Power and thermal limits, GPU downclock
- GPU Boost which automatically adjusts the core clock within power and thermal limits.
- GPUs may be throttled due to excessive heat caused by previous runs


T - lock the clocks for consistency so that the GPU always runs at a fixed maximum frequency.

A

lock the clocks for consistency so that the GPU always runs at a fixed maximum frequency.

R

run-to-run performance is stable and not subject to variations in power or temperature. 


#### S - No workload/idle, GPU sleep
T - Persistence mode that keeps the GPU driver loaded and the hardware in a ready state even when no application is active. 

A

nvidia-smi -pm 1

R

shaves off job-startup latency and prevents cold start delays


#### S - bit memory error
- For long training or inference jobs on huge models, a single memory error could crash the job completely or, even worse, silently corrupt your model without a warning. 

T - Error Correcting Code (ECC) memory on GPUs

- if there’s a single-bit memory error caused by cosmic rays, ensure the memory can be corrected on the fly
- if there’s a double-bit error, the error is detected and will throw an error to the calling code. 
- ECC is always enabled and cannot be disabled. 

R

memory-error protection ensure reliability
