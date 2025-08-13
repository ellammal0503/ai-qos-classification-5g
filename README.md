  ~\_  ####_        Amazon Linux 2023
  ~~  \_#####\
  ~~     \###|
  ~~       \#/ ___   https://aws.amazon.com/linux/amazon-linux-2023
   ~~       V~' '->
    ~~~         /
      ~~._.   _/
         _/ _/
       _/m/'
Last login: Tue Aug 12 16:26:11 2025 from 18.206.107.29
[ec2-user@ip-172-31-35-67 ~]$ ls
Multimodel_optimized
[ec2-user@ip-172-31-35-67 ~]$ em -rf *
-bash: em: command not found
[ec2-user@ip-172-31-35-67 ~]$ rm -rf *
[ec2-user@ip-172-31-35-67 ~]$ ls
[ec2-user@ip-172-31-35-67 ~]$  git clone https://github.com/ellammal0503/Multimodel_optimized.git
Cloning into 'Multimodel_optimized'...
remote: Enumerating objects: 93, done.
remote: Counting objects: 100% (93/93), done.
remote: Compressing objects: 100% (77/77), done.
remote: Total 93 (delta 20), reused 89 (delta 16), pack-reused 0 (from 0)
Receiving objects: 100% (93/93), 4.88 MiB | 36.18 MiB/s, done.
Resolving deltas: 100% (20/20), done.
[ec2-user@ip-172-31-35-67 ~]$ ls
Multimodel_optimized
[ec2-user@ip-172-31-35-67 ~]$ cd Multimodel_optimized/
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ ls
Dockerfile   docker-compose.yml  main.py  models              requirements.txt            schema.py                           train_all_models.py
__pycache__  encoders            metrics  qos-classification  response_1755010146890.txt  synthetic_5g_qos_dataset_10000.csv  utils.py
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ cat /etc/os-release
NAME="Amazon Linux"
VERSION="2023"
ID="amzn"
ID_LIKE="fedora"
VERSION_ID="2023"
PLATFORM_ID="platform:al2023"
PRETTY_NAME="Amazon Linux 2023.8.20250808"
ANSI_COLOR="0;33"
CPE_NAME="cpe:2.3:o:amazon:amazon_linux:2023"
HOME_URL="https://aws.amazon.com/linux/amazon-linux-2023/"
DOCUMENTATION_URL="https://docs.aws.amazon.com/linux/"
SUPPORT_URL="https://aws.amazon.com/premiumsupport/"
BUG_REPORT_URL="https://github.com/amazonlinux/amazon-linux-2023"
VENDOR_NAME="AWS"
VENDOR_URL="https://aws.amazon.com/"
SUPPORT_END="2029-06-30"
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ docker

Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx (Docker Inc., 0.12.1)
  checkpoint  Manage checkpoints
  container   Manage containers
  context     Manage contexts
  image       Manage images
  manifest    Manage Docker image manifests and manifest lists
  network     Manage networks
  plugin      Manage plugins
  system      Manage Docker
  trust       Manage trust on Docker images
  volume      Manage volumes

Swarm Commands:
  config      Manage Swarm configs
  node        Manage Swarm nodes
  secret      Manage Swarm secrets
  service     Manage Swarm services
  stack       Manage Swarm stacks
  swarm       Manage Swarm

Commands:
  attach      Attach local standard input, output, and error streams to a running container
  commit      Create a new image from a container's changes
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  diff        Inspect changes to files or directories on a container's filesystem
  events      Get real time events from the server
  export      Export a container's filesystem as a tar archive
  history     Show the history of an image
  import      Import the contents from a tarball to create a filesystem image
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  load        Load an image from a tar archive or STDIN
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Global Options:
      --config string      Location of client config files (default "/home/ec2-user/.docker")
  -c, --context string     Name of the context to use to connect to the daemon (overrides DOCKER_HOST env var and default context set with "docker
                           context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info", "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default "/home/ec2-user/.docker/ca.pem")
      --tlscert string     Path to TLS certificate file (default "/home/ec2-user/.docker/cert.pem")
      --tlskey string      Path to TLS key file (default "/home/ec2-user/.docker/key.pem")
      --tlsverify          Use TLS and verify the remote
  -v, --version            Print version information and quit

Run 'docker COMMAND --help' for more information on a command.

For more help on how to use Docker, head to https://docs.docker.com/go/guides/
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$  sudo docker run -d -p 8000:8000 qos-classifier-api
ee0b2aa6f1f31a6ef390263c86aa3a3fb450c25e005145fd6699d28d5ff5fd17
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ docker -ps
unknown shorthand flag: 'p' in -ps
See 'docker --help'.

Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx (Docker Inc., 0.12.1)
  checkpoint  Manage checkpoints
  container   Manage containers
  context     Manage contexts
  image       Manage images
  manifest    Manage Docker image manifests and manifest lists
  network     Manage networks
  plugin      Manage plugins
  system      Manage Docker
  trust       Manage trust on Docker images
  volume      Manage volumes

Swarm Commands:
  config      Manage Swarm configs
  node        Manage Swarm nodes
  secret      Manage Swarm secrets
  service     Manage Swarm services
  stack       Manage Swarm stacks
  swarm       Manage Swarm

Commands:
  attach      Attach local standard input, output, and error streams to a running container
  commit      Create a new image from a container's changes
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  diff        Inspect changes to files or directories on a container's filesystem
  events      Get real time events from the server
  export      Export a container's filesystem as a tar archive
  history     Show the history of an image
  import      Import the contents from a tarball to create a filesystem image
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  load        Load an image from a tar archive or STDIN
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Global Options:
      --config string      Location of client config files (default "/home/ec2-user/.docker")
  -c, --context string     Name of the context to use to connect to the daemon (overrides DOCKER_HOST env var and default context set with "docker
                           context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info", "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default "/home/ec2-user/.docker/ca.pem")
      --tlscert string     Path to TLS certificate file (default "/home/ec2-user/.docker/cert.pem")
      --tlskey string      Path to TLS key file (default "/home/ec2-user/.docker/key.pem")
      --tlsverify          Use TLS and verify the remote
  -v, --version            Print version information and quit

Run 'docker COMMAND --help' for more information on a command.

For more help on how to use Docker, head to https://docs.docker.com/go/guides/

[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ docker -s
unknown shorthand flag: 's' in -s
See 'docker --help'.

Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx (Docker Inc., 0.12.1)
  checkpoint  Manage checkpoints
  container   Manage containers
  context     Manage contexts
  image       Manage images
  manifest    Manage Docker image manifests and manifest lists
  network     Manage networks
  plugin      Manage plugins
  system      Manage Docker
  trust       Manage trust on Docker images
  volume      Manage volumes

Swarm Commands:
  config      Manage Swarm configs
  node        Manage Swarm nodes
  secret      Manage Swarm secrets
  service     Manage Swarm services
  stack       Manage Swarm stacks
  swarm       Manage Swarm

Commands:
  attach      Attach local standard input, output, and error streams to a running container
  commit      Create a new image from a container's changes
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  diff        Inspect changes to files or directories on a container's filesystem
  events      Get real time events from the server
  export      Export a container's filesystem as a tar archive
  history     Show the history of an image
  import      Import the contents from a tarball to create a filesystem image
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  load        Load an image from a tar archive or STDIN
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Global Options:
      --config string      Location of client config files (default "/home/ec2-user/.docker")
  -c, --context string     Name of the context to use to connect to the daemon (overrides DOCKER_HOST env var and default context set with "docker
                           context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info", "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default "/home/ec2-user/.docker/ca.pem")
      --tlscert string     Path to TLS certificate file (default "/home/ec2-user/.docker/cert.pem")
      --tlskey string      Path to TLS key file (default "/home/ec2-user/.docker/key.pem")
      --tlsverify          Use TLS and verify the remote
  -v, --version            Print version information and quit

Run 'docker COMMAND --help' for more information on a command.

For more help on how to use Docker, head to https://docs.docker.com/go/guides/

[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                       NAMES
ee0b2aa6f1f3   qos-classifier-api   "uvicorn main:app --…"   32 seconds ago   Up 31 seconds   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   sleepy_swirles
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker build -t qos-classifier-api .
[+] Building 114.6s (10/10) FINISHED                                                                                                        docker:default
 => [internal] load build definition from Dockerfile                                                                                                  0.0s
 => => transferring dockerfile: 478B                                                                                                                  0.0s
 => [internal] load metadata for docker.io/library/python:3.9-slim                                                                                    0.3s
 => [internal] load .dockerignore                                                                                                                     0.0s
 => => transferring context: 2B                                                                                                                       0.0s
 => [1/5] FROM docker.io/library/python:3.9-slim@sha256:dc5f8a552b07c0b7edaf14a3ced4ca78e308b0701c7ff125d229644190094552                              2.8s
 => => resolve docker.io/library/python:3.9-slim@sha256:dc5f8a552b07c0b7edaf14a3ced4ca78e308b0701c7ff125d229644190094552                              0.0s
 => => sha256:396b1da7636e2dcd10565cb4f2f952cbb4a8a38b58d3b86a2cacb172fb70117c 29.77MB / 29.77MB                                                      0.4s
 => => sha256:0219e1e5e6ef3ef9d91f78826576a112b1c20622c10c294a4a105811454d1cb1 1.29MB / 1.29MB                                                        0.1s
 => => sha256:5ec99fe17015e703c289d110b020e4e362d5b425be957d68bfb400d56d83f234 13.37MB / 13.37MB                                                      0.3s
 => => sha256:dc5f8a552b07c0b7edaf14a3ced4ca78e308b0701c7ff125d229644190094552 9.08kB / 9.08kB                                                        0.0s
 => => sha256:213766eae7e1ad5da6140428e7f15db89f2c83caf906cc06fc9c5c8a0028e3b6 1.74kB / 1.74kB                                                        0.0s
 => => sha256:28f8802246faa922c08dd76e3ec467e3cb4278af72e99e1afa2f68dfb9ea991d 5.30kB / 5.30kB                                                        0.0s
 => => sha256:ea3499df304f0a84e9f076a05f0cfe2a64d8fcb884894ce682df9204c6a18a91 249B / 249B                                                            0.2s
 => => extracting sha256:396b1da7636e2dcd10565cb4f2f952cbb4a8a38b58d3b86a2cacb172fb70117c                                                             1.3s
 => => extracting sha256:0219e1e5e6ef3ef9d91f78826576a112b1c20622c10c294a4a105811454d1cb1                                                             0.1s
 => => extracting sha256:5ec99fe17015e703c289d110b020e4e362d5b425be957d68bfb400d56d83f234                                                             0.7s
 => => extracting sha256:ea3499df304f0a84e9f076a05f0cfe2a64d8fcb884894ce682df9204c6a18a91                                                             0.0s
 => [internal] load build context                                                                                                                     0.2s
 => => transferring context: 12.96MB                                                                                                                  0.1s
 => [2/5] WORKDIR /app                                                                                                                                0.3s
 => [3/5] COPY . .                                                                                                                                    0.1s
 => [4/5] RUN pip install --upgrade pip                                                                                                               4.0s
 => [5/5] RUN pip install -r requirements.txt                                                                                                        75.5s 
 => exporting to image                                                                                                                               31.5s 
 => => exporting layers                                                                                                                              31.5s 
 => => writing image sha256:caded5c00a6b7fd41b2d59801168566d1d69192201c792bcc679c4894f3d53c2                                                          0.0s 
 => => naming to docker.io/library/qos-classifier-api                                                                                                 0.0s 
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$  sudo docker run -d -p 8000:8000 qos-classifier-api
db58f2d090bc9de8cddad395d6a894565d561ccfb6f721f212f0d18f3a6b8377
docker: Error response from daemon: driver failed programming external connectivity on endpoint charming_moore (f24a75ef81e48d65515aa5fc9fc558a551a6af90bfab971b759c85a4c2f96bb3): Bind for 0.0.0.0:8000 failed: port is already allocated.
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS                                       NAMES
ee0b2aa6f1f3   c2ee369791b4   "uvicorn main:app --…"   6 minutes ago   Up 6 minutes   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   sleepy_swirles
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker stop gifted_hamilton
Error response from daemon: No such container: gifted_hamilton
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS                                       NAMES
ee0b2aa6f1f3   c2ee369791b4   "uvicorn main:app --…"   7 minutes ago   Up 7 minutes   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   sleepy_swirles
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker stop sleepy_swirles
sleepy_swirles
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker rm sleepy_swirles
sleepy_swirles
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$  sudo docker run -d -p 8000:8000 qos-classifier-api
3fbd6cd83608eeae05946cfa1a3b1ea9231b352ebfeb30726709ed658ef90ca5
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                                       NAMES
3fbd6cd83608   qos-classifier-api   "uvicorn main:app --…"   4 seconds ago   Up 3 seconds   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   practical_clarke
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                                       NAMES
3fbd6cd83608   qos-classifier-api   "uvicorn main:app --…"   2 minutes ago   Up 2 minutes   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   practical_clarke
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ aws ec2 describe-instances \
  --query "Reservations[].Instances[].SecurityGroups[]" \
  --output table

Unable to locate credentials. You can configure credentials by running "aws configure".

[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ 
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ aws ec2 describe-instances 

Unable to locate credentials. You can configure credentials by running "aws configure".
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                       NAMES
3fbd6cd83608   qos-classifier-api   "uvicorn main:app --…"   13 minutes ago   Up 13 minutes   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   practical_clarke
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker stop practical_clarke
practical_clarke
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker rm practical_clarke
practical_clarke
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ sudo docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
[ec2-user@ip-172-31-35-67 Multimodel_optimized]$ 
