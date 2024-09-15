from azure.ai.ml.automl import ClassificationPrimaryMetrics, automl, ml_client
 
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

#print(list(ClassificationPrimaryMetrics))

"""
[<ClassificationPrimaryMetrics.AUC_WEIGHTED: 'AUCWeighted'>, <ClassificationPrimaryMetrics.ACCURACY: 'Accuracy'>, 
<ClassificationPrimaryMetrics.NORM_MACRO_RECALL: 'NormMacroRecall'>, 
<ClassificationPrimaryMetrics.AVERAGE_PRECISION_SCORE_WEIGHTED: 'AveragePrecisionScoreWeighted'>,
 <ClassificationPrimaryMetrics.PRECISION_SCORE_WEIGHTED: 'PrecisionScoreWeighted'>]
"""


my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")
 
classification_job = automl.classification(
    compute="christava1",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)

classification_job.set_limits(
    timeout_minutes=60, 
    trial_timeout_minutes=20, 
    max_trials=5,
    enable_early_termination=True,
)

returned_job = ml_client.jobs.create_or_update(
    classification_job
)

aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
