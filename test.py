import matplotlib.pyplot as plt
from OnePick import OnePick
import boto3
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

accessKey = os.getenv('AWS_ACCESS_KEY')
secretKey = os.getenv('AWS_SECRET_KEY')
region = os.getenv('AWS_REGION')
bucket_name = os.getenv('AWS_BUCKET_NAME')

user_id = '2'
query ='자동차'



test = OnePick(user_id)
print(test.preproces_and_save())

top_list = test.getImagePath(query)



def render_image(image_paths):
  session = boto3.Session(
  aws_access_key_id=accessKey,
  aws_secret_access_key=secretKey,
  region_name=region
  )
  s3 = session.client('s3')
  # Getting images
  images=[]
  for key in image_paths:
    # Get the object
    obj_data = s3.get_object(Bucket=bucket_name, Key=key)
    # Read the object (image file) data into a stream
    image_stream = BytesIO(obj_data['Body'].read())
    # Load the image into a PIL.Image object and display it
    image = Image.open(image_stream)
    # append Image to Images
    images.append(image)
  fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 10))  # Adjust the figure size here
  # Display each image in a subplot
  for i, image in enumerate(images):
      axes[i].imshow(image)
      axes[i].axis('off')  # Hide axes
  plt.show()

render_image(top_list)

