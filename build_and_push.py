import subprocess
import sys


get_login = True
acc_id = 'aws_account_id'
iam_user = 'iam_user'
region = 'aws_region'
mode = 'cpu' # set to 'cpu' or 'gpu'

login = f'$(aws ecr get-login --no-include-email --region {region})'
build = ['docker', 'build', '-f', f'Dockerfile.{mode}', '-t', mode, '.']
tag = ['docker', 'tag', f'{mode}:latest', f'{acc_id}.dkr.ecr.{region}.amazonaws.com/{iam_user}:{mode}']
push = ['docker', 'push', f'{acc_id}.dkr.ecr.{region}.amazonaws.com/{iam_user}:{mode}']

exit_code = 0

if get_login:
    exit_code |= subprocess.run(login, shell=True).returncode

for cmd in [build, tag]:
    exit_code |= subprocess.run(cmd).returncode

sys.exit(exit_code)
