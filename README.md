# Rysia Benchmarking Framework
Rysia is a declartive benchmarking framework for deep learning systems. Users specify deep learning workloads in 
blueprint files without any coding required. Compilation into platform specific execution is done automatically.
Benchmarking experiments can be executed locally or in an AWS cloud environment.
***
### Installation
To install CPU-restricted packages, run 
```bash
python setup_cpu.py install
```

To install GPU-accelerated packages, run 
```bash
python setup_gpu.py install
``` 
***
### Usage
Experiment worklaods are specified fully declarative in blueprint files which are implemented as Python modules. 
Examples can be found in the ```blueprints``` folder. An overview over available parameters can be found in 
the ```blueprint_example.py``` file.


#### Local Execution
To execute benchmarking workloads locally:
1. Install the package
2. Specify a blueprint file and set the parameter ```aws``` to False
3. Run ```rysia blueprint.py```

#### AWS Execution
To run benchmarking workloads within the AWS cloud environment:
1. Install the awscli and configure a valid AWS account
2. Build and push the package to an Amazon ECR repository by 
specifying the required parameters in ```build_and_push.py``` and then running
```python build_and_push.py```
3. Specify job definitions that are linked to the
ECR container images in the AWS Batch web terminal
4. Specify a blueprint file and set the parameter ```aws``` to True
5. Run ```rysia blueprint.py```
***
