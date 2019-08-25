'''
import deep learning model DSL
'''
from rysia.core import blueprint as bp

'''
sequence of model layers. Check other blueprint files for examples
'''
architecture = bp.model


'''
learning rate: value between 0 and 1
'''
learning_rate = float

'''
number of training epochs. Each epoch passes over 
the entire dataset with one iteration per mini-batch
'''
epochs = int

'''
mini-batch size used in each training iteration
'''
batch_size = int

'''
available loss functions
{'cross_entropy', 'mean_squared_error'}
'''
loss_function = str

'''
available optimizer
{'adam', 'sgd'}
'''
optimizer = str

'''
choose recurrent when benchmarking recurrent neural networks
choose 'default' otherwise
{'default', 'recurrent'} 
'''
optimizer_mode = str

'''
list of LSTM state sizes 
(set to none when training non-recurrent networks)
'''
state_sizes = [int]

'''
length of training sequences 
(set to none when benchmarking non-recurrent networks)
'''
train_seq_length = int

'''
length of test sequences 
(set to none when benchmarking non-recurrent networks)
'''
test_seq_length = int

'''
sequence cut-off length for recurrent backpropagation 
(set to none when benchmarking non-recurrent networks)
'''
truncated_backprop_length = int

'''
reshape input data to given shape 
(set to [None] if shape should remain as is)
'''
reshape_input_data = [int]

'''
number of repetitions for each combination of
software platform and hardware environment
'''
runs = int

'''
python random seed
'''
python_seed = int

'''
numpy random seed
'''
numpy_seed = int

'''
framework random seed
'''
framework_seed = int

'''
hardware metrics to monitor
{'gpu', 'cpu', 'memory', 'diskIO'}
'''
monitors = [str]

'''
benchmark training or inference workloads
{'training', 'inference'}
'''
mode = str

'''
list of software platforms to benchmark
{'mxnet', 'tensorflow', 'pytorch'}
'''
frameworks = [str]


'''
set True when running on AWS, set False when running locally 
(all following AWS parameter will be ignored)
'''
aws = bool

'''
name of AWS region
'''
region_name = str

'''
name of S3 bucket that contains datasets and stores results
'''
bucket_name = str

'''
list of EC2 instances to execute workloads
'''
instance_types = [str]

'''
AWS Batch job name
'''
job_name = str

'''
ID of Amazon Machine Image. 
Required when loading custom image that utilizes GPU functionality
'''
ami_id = str

'''
min CPU cores for AWS Batch compute environment
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
env_min_cpu = int

'''
desired CPU cores for AWS Batch compute environment
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
env_desired_cpu = int

'''
max CPU cores for AWS Batch compute environment
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
env_max_cpu = int

'''
list of used Amazon EC2 subnets
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
subnets = [str]

'''
list of Amazon EC2 security group ids
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
security_group_ids = [str]

'''
AWS Batch instance role
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
instance_role = str

'''
AWS Batch service role
for more details see
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.create_compute_environment
'''
service_role = str

'''
AWS account id
'''
account_id = str

'''
list of container images stored in Amazon ECR
should contain one image for cpu and one for gpu execution
'''
container_images = [str]

'''
list of AWS Batch job definitions
should contain one definition for cpu and one for gpu execution
that are linked to the corresponding container images above
'''
job_def_names = [str]

'''
number of CPU cores required by AWS Batch job
(set to None for default amount of available CPU cores)
'''
job_num_vcpu = None

'''amount of memory required by AWS Batch job
(set to None for maximum memory available'''
job_memory_size = None

'''
set True for deleting compute environment after AWS Batch job finishes
'''
tear_down_comp_env = False


'''
location of training data 
local file path or Amazon S3 key
must point to Numpy array (.npy file)
'''
train_data_path = str

'''
location of training label
local file path or Amazon S3 key
must point to Numpy array (.npy file)
'''
train_label_path = str

'''
location of test data
local file path or Amazon S3 key
must point to Numpy array (.npy file)
'''
test_data_path = str

'''
location of test label
local file path or Amazon S3 key
must point to Numpy array (.npy file)
'''
test_label_path = str

'''
location to store benchmarking results
local file path
'''
result_folder_path = str

'''
location of model parameters
local file path or Amazon S3 key
must point to Python pickle file
'''
model_params_path = str
