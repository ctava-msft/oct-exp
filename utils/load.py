import mltable

# glob the parquet file paths for years 2015-19, all months.
paths = [
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2015/puMonth=*/*.parquet"
    }
]

"""
paths = [
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2015/puMonth=*/*.parquet"
    },
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2016/puMonth=*/*.parquet"
    },
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2017/puMonth=*/*.parquet"
    },
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2018/puMonth=*/*.parquet"
    },
    {
        "pattern": "wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=2019/puMonth=*/*.parquet"
    },
] """

# create a table from the parquet paths
tbl = mltable.from_parquet_files(paths)

# table a random sample
tbl = tbl.take_random_sample(probability=0.001, seed=735)

# filter trips with a distance > 0
tbl = tbl.filter("col('tripDistance') > 0")

# Drop columns
tbl = tbl.drop_columns(["puLocationId", "doLocationId", "storeAndFwdFlag"])

# Create two new columns - year and month - where the values are taken from the path
tbl = tbl.extract_columns_from_partition_format("/puYear={year}/puMonth={month}")

# print the first 5 records of the table as a check
tbl.show(5)

# save the data loading steps in an MLTable file to a cloud storage resource
subscription_id = "1c47c29b-10d8-4bc6-a024-05ec921662cb"
resource_group = "christava-rg"
workspace_name = "train"
datastore_name = "workspacefilestore"
table_name="nyctaxi"
tbl.save(path=f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/{datastore_name}", colocated=True, show_progress=True, overwrite=True)