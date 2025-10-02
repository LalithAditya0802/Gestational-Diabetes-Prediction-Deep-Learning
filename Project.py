# In[1]:
import numexpr
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
import sagemaker
from sagemaker import RandomCutForest
from sagemaker.serializers import CSVSerializer
import sagemaker.amazon.common as smac
# from sagemaker.amazon.record_pb2 import CounterExample
# from sagemaker.amazon.amazon_estimator import RecordEncoder
# In[2]:
from sagemaker.session import s3_input, Session
# In[3]:
bucket_name = 'diabetesNaniPredictions' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME
FOR YOUR BUCKET
my_region = boto3.session.Session().region_name # set the region of the instance
print(my_region)
30
# In[4]:
AWS_REGION = "ap-south-1"
client = boto3.client("s3", region_name=AWS_REGION)
bucket_name = "diabetesnanipredictionsnani"
location = {'LocationConstraint': AWS_REGION}
response = client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
print("Amazon S3 bucket has been created")
# In[ ]:
s3 = boto3.resource('s3')
# try:
# if my_region == 'ap-south-1':
# s3.create_bucket(Bucket=bucket_name)
# print('S3 bucket created successfully')
# except Exception as e:
# print('S3 error: ',e)
# In[ ]:
# set an output path where the trained model will be saved
prefix = 'random-forest-as-a-built-in-algo'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)
# #### Downloading The Dataset And Storing in S3
# In[ ]:
import pandas as pd
import urllib
# try:
# urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learningmodel-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
# print('Success: downloaded bank_clean.csv.')
# except Exception as e:
# print('Data load error: ',e)
try:
data = pd.read_csv('./diabetesdata.csv')
31
print('Success: Data loaded into dataframe.')
except Exception as e:
print('Data load error: ',e)
# In[8]:
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# In[9]:
data
# In[10]:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["class"] = le.fit_transform(data["class"])
data["class"]
# In[11]:
data
# In[12]:
#extracting numerical columns values
x_independent = data.iloc[:,:-1]
y_dependent=data.iloc[:,8:9]
# In[13]:
y_dependent
# In[14]:
name=x_independent.columns
name
# In[15]:
names2 =y_dependent.columns
names2
32
# In[16]:
#Normalisation
from sklearn.preprocessing import MinMaxScaler
# In[17]:
scale=MinMaxScaler()
# In[18]:
X_scaled=scale.fit_transform(x_independent)
# In[19]:
X=pd.DataFrame(X_scaled,columns=name)
# In[20]:
y_dependent=pd.DataFrame(y_dependent,columns=names2)
# In[21]:
y_dependent
# In[22]:
X
# In[23]:
x=X #independent values
y=y_dependent
# In[24]:
y
# In[25]:
d_new = pd.concat([x,y],axis=1)
data = d_new
# In[26]:
data
33
# In[27]:
### Train Test split
import numpy as np
train_data, test_data = np.split(data.sample(frac=1, random_state=38), [int(0.75 * len(data))])
print(train_data.shape, test_data.shape)
# In[28]:
test_data
# In[29]:
### Saving Train And Test Into Buckets
## We start with Train Data
import os
pd.concat([train_data['class'], train_data.drop(['class'],
axis=1)],
axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,
'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix),
content_type='text/csv', distribution='ShardedByS3Key')
# In[30]:
# Test Data Into Buckets
pd.concat([test_data['class'], test_data.drop(['class'], axis=1)], axis=1).to_csv('test.csv', index=False,
header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,
'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix),
content_type='text/csv', distribution='ShardedByS3Key')
# ### Building Models RandomcutForest- Inbuilt Algorithm
# In[31]:
# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
34
# container = get_image_uri(boto3.Session().region_name,
# 'xgboost',
# repo_version='1.0-1')
# In[32]:
from sagemaker.amazon.amazon_estimator import image_uris
container = image_uris.retrieve(region=boto3.Session().region_name, framework='randomcutforest',
version='1.0-1')
# In[33]:
# # Initialize hyperparameters for multi-class classification
# hyperparameters = {
# "max_depth": "5",
# "eta": "0.2",
# "gamma": "4",
# "min_child_weight": "6",
# "subsample": "0.7",
# "objective": "multi:softmax", # Set objective to multi-class softmax
# "num_class": 8, # Number of classes in your output column
# "num_round": 50
# }
# In[34]:
# define the hyperparameters for
hyperparameters = {
'num_trees': 100, # Number of trees in the forest
'eval_metrics': 'accuracy', # Evaluation metric
'feature_dim': 8 # number of features in your training data
}
# In[35]:
# # construct a SageMaker estimator that calls the xgboost-container
# estimator = sagemaker.estimator.Estimator(image_uri=container, #
hyperparameters=hyperparameters,
# role=sagemaker.get_execution_role(),
# instance_count=1,
# instance_type='ml.m5.2xlarge',
35
# volume_size=5, # 5 GB
# output_path=output_path,
# use_spot_instances=True,
# max_run=300,
# max_wait=600)
# In[36]:
# construct a SageMaker estimator
estimator = sagemaker.estimator.Estimator(
role=sagemaker.get_execution_role(),
instance_count=1,
instance_type='ml.m5.2xlarge',
volume_size=5,
output_path=output_path,
use_spot_instances=True,
max_run=300,
max_wait=600,
hyperparameters=hyperparameters,
image_uri=container)
# In[37]:
estimator.fit({'train': s3_input_train})
# ### Deploy Machine Learning Model As Endpoints
# In[38]:
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
# #### Prediction of the Test Data
# In[39]:
pip install --upgrade sagemaker
# In[40]:
from sagemaker.serializers import CSVSerializer
test_data_array = test_data.drop(['class'], axis=1).values #load the data into an array
36
xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
# In[43]:
result=xgb_predictor.predict([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
result
# In[ ]:
result.decode('utf-8')
# In[ ]:
result[]
predictions_array
# In[ ]:
cm = pd.crosstab(index=test_data['NObeyesdad'], columns=np.round(predictions_array),
rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p =
(tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn,
fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn,
tp/(tp+fp)*100, tp))
37
# #### Deleting The Endpoints
# In[ ]:
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()