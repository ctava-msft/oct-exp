import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = "1c47c29b-10d8-4bc6-a024-05ec921662cb"
resource_group = "christava-rg"
workspace = "train"
datastore_name = "workspacefilestore"
table_name="MLTable"

# connect to the AzureML workspace
# NOTE: the subscription_id, resource_group, workspace variables are set
# in a previous code snippet.
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# get the latest version of the data asset
# Note: The version was set in the previous snippet. If you changed the version
# number, update the VERSION variable below.
#VERSION="1"
#data_asset = ml_client.data.get(name=table_name, version=VERSION)

url="azureml://subscriptions/1c47c29b-10d8-4bc6-a024-05ec921662cb/resourcegroups/christava-rg/workspaces/train/datastores/workspacefilestore/paths/MLTable"

#tbl = mltable.load(f"azureml:/{data_asset.id}")
tbl = mltable.load(url)
tbl.show(5)

# load into pandas
# NOTE: The data is in East US region and the data is large, so this will take several minutes (~7mins) to load if you are in a different region.
#df = tbl.to_pandas_dataframe()