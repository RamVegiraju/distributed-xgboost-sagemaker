## [Blog](https://towardsdatascience.com/training-xgboost-on-a-1tb-dataset-8790e2bc8672)

## XGBoost Distributed Training 1TB

For this example we'll be scaling the [Abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone) to 1TB size and training the [SageMaker XGBoost algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) on it.

## Data Upload

We will be uploading the dataset to S3, for our development we will utilize EC2. Our dataset itself won't be 1TB rather 10,000 100 MB files that we will upload to an S3 bucket accumulating into 1TB. This works a lot better with the SageMaker Training via [Fast File Mode](https://aws.amazon.com/about-aws/whats-new/2021/10/amazon-sagemaker-fast-file-mode/). To understand the different input data modes for SageMaker Training read this [blog](https://aws.amazon.com/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/).

To create this dataset kick of an [EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html), we use an r6a.48xlarge instance which has more memory to handle larger sets of data and any computations. Once you have logged into the instance configure the AWS CLI and run the following command to retreive the public Abalone dataset.

```
aws configure #provide authentication/credentials
```

```
#retreive data
aws s3 cp s3://sagemaker-sample-files/datasets/tabular/uci_abalone/train_csv/abalone_dataset1_train.csv .
```

The initial dataset is around 100KB we enlarge it to 100MB so we can upload less copies to S3 and also as this size distribution is better suited for FastFile Mode.

```
import os
import pandas as pd
import sys

#~110KB initial file
df = pd.read_csv("abalone_dataset1_train.csv")
print(sys.getsizeof(df))

#creates a 104MB file
df_larger = pd.concat([df]*700, ignore_index=True)
print(sys.getsizeof(df_larger))

df_larger.to_csv("abalone-100mb.csv")
```

We can then take this dataset and upload 10,000 copies to S3 to achieve 1TB size. Once again you can splice your dataset as needed it does not need to follow the specific recipe of this example. To speed up this operation I would also recommend using some multiprocessing Python code to upload the data at a faster rate, this should currently take ~2 hours.

```
%%sh

#replace with your S3 bucket to upload to
s3_bucket='sagemaker-us-east-1-474422712127'

for i in {0..10000}
do
  aws s3 cp abalone-100mb.csv s3://$s3_bucket/xgboost-1TB/abalone-$i.csv 
done
```

## Training Job

For training we will use the SageMaker Python SDK to define our input and estimator. First we define our TrainingInput which will define the key parameters of the input mode and as well as our training path.

```
from sagemaker.inputs import TrainingInput

#replace with your S3 Bucket with data
training_path = 's3://sagemaker-us-east-1-474422712127/xgboost-1TB/'

#set distribution to ShardedByS3Key otherwise a copy of all files will be made across all instances
train_input = TrainingInput(training_path, content_type="text/csv", input_mode='FastFile', distribution = "ShardedByS3Key")
training_path
```

We then define our instance type and count, in this case we use 25 ml.m5.24xlarge instances you can tune this to decrease time or cost as needed. With the current setup it takes about 11 hours.

```
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=25,
    output_path=model_path,
    sagemaker_session=sagemaker_session,
    role=role,
)
``` 
