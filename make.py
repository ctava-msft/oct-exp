import os
import json
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Define connection details
account_name = "redacted"
account_key = "<redacted>"
container_name = "redacted"
datastore_name = "workspaceblobstore"
subscription_id = "<redacted>"
resource_group = "rg-redacted"
workspace = "redacted"

# Connect to the Azure Blob Storage account
blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
container_client = blob_service_client.get_container_client(container_name)
print(container_client)

# List blobs in the container
blobs_list = container_client.list_blobs()
print(blobs_list)
#blobs_list = "oct"

# Define the JSONL file path
jsonl_file_path = "data.jsonl"

# Write the JSONL data to a file
with open(jsonl_file_path, 'w') as jsonl_file:
    for blob in blobs_list:
        print(blob)
        # Assuming the blob name is the image file name and the label is derived from the file name
        image_url = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}/paths/{blob.name}"
        image_format = blob.name.split('.')[-1]
        label = blob.name.split('/')[-1].split('-')[0].lower()  # Assuming label is the first part of the file name before '_'
        
        # Placeholder values for width and height
        width = "512px"
        height = "512px"
        
        json_object = {
            "image_url": image_url,
            "image_details": {
                "format": image_format,
                "width": width,
                "height": height
            },
            "label": label
        }
        
        jsonl_file.write(json.dumps(json_object) + '\n')

print(f"File {jsonl_file_path} generated based on entries in the container {container_name}.")