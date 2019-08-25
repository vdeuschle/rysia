from rysia.core import blueprint as bp

architecture = bp.model(
    bp.InputLayer(shape=(250, 50)),
    bp.LSTM(size=128, return_full_seq=False),
    bp.FullyConnected(size=2, activation='linear')
)

learning_rate = 0.01
epochs = 100
batch_size = 128
loss_function = 'cross_entropy'
optimizer = 'adam'
optimizer_mode = 'recurrent'
state_sizes = [128]
train_seq_length = 250
test_seq_length = 250
truncated_backprop_length = 250
reshape_input_data = [None]

runs = 7
python_seed = 42
numpy_seed = 42
framework_seed = 42
monitors = []
mode = 'training'
frameworks = ['mxnet', 'tensorflow', 'pytorch']

aws = True
region_name = 'aws_region'
bucket_name = 'bucket_name'
instance_types = ['c4.2xlarge', 'c5.2xlarge', 'p2.xlarge', 'p3.2xlarge']
job_name = 'imdb'
ami_id = 'ami_id'
env_min_cpu = 0
env_desired_cpu = 0
env_max_cpu = 512
subnets = ['subnet1', 'subnet2', 'subnet3']
security_group_ids = ['secourity_group_id']
instance_role = 'instance_role_arn'
service_role = 'service_role_arn'
account_id = 'aws_account_id'
container_images = ['cpu_image', 'gpu_image']
job_def_names = ['cpu_job_definition', 'gpu_job_definition']
job_num_vcpu = None
job_memory_size = None
tear_down_comp_env = False

train_data_path = 'train_data_path'
train_label_path = 'train_label_path'
test_data_path = None
test_label_path = None
result_folder_path = 'result_folder_path'
model_params_path = None
