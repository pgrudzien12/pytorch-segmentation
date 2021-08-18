from azureml.core.authentication import InteractiveLoginAuthentication

# Wymagany jest element azureml-core w wersji 1.0.72 lub nowszej
from azureml.core import Workspace, Dataset
interactive_auth = InteractiveLoginAuthentication(tenant_id="f76b0fd4-dd1a-422a-9c43-43d9f41f789f")

subscription_id = '319debd4-fdd7-4225-9a05-3b00076ea0cb'
resource_group = 'mlops'
workspace_name = 'mlops'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='roadsegmv1')
dataset.download(target_path='./data/img', overwrite=False)