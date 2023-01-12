%%sh

#replace with your S3 bucket to upload to
s3_bucket='sagemaker-us-east-1-474422712127'

for i in {0..10000}
do
  aws s3 cp abalone-100mb.csv s3://$s3_bucket/xgboost-1TB/abalone-$i.csv 
done
