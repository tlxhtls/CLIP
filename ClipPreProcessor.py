import torch
import clip
import os
from PIL import Image
from io import BytesIO
import boto3, botocore
from dotenv import load_dotenv
load_dotenv()

accessKey = os.getenv('AWS_ACCESS_KEY')
secretKey = os.getenv('AWS_SECRET_KEY')
region = os.getenv('AWS_REGION')
bucket_name = os.getenv('AWS_BUCKET_NAME')

class ClipPreProcessor:

    def __init__(self, user_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.user_id = user_id
        self.s3 = self.start_s3()


    def start_s3(self):
        # Create a session using your AWS credentials
        session = boto3.Session(
          aws_access_key_id=accessKey,
          aws_secret_access_key=secretKey,
          region_name=region
        )
        s3 = session.client('s3', config=botocore.client.Config(read_timeout=180))
        #사진이 많을 경우 s3에서 사진을 불러오는 과정에서 ReadTimeoutError가 발생할 수 있습니다. read_timeout을 크게 설정하면 해당 에러를 방지할 수 있습니다.
        return s3

    def get_image_paths(self):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        prefix = f'{self.user_id}/'
        s3 = self.s3
        # Get the list of objects in the directory
        objects = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        # Make sure objects were found
        if 'Contents' in objects:
            # Getting image_paths
            paths=[obj['Key'] for obj in objects['Contents']]
            # Filter only image files
            image_paths = [path for path in paths if any(path.lower().endswith(ext) for ext in extensions)]
        else:
              print("No images found.")
        return image_paths

    def get_images(self, image_paths):
        # Getting images
        images=[]
        for key in image_paths:
          # Get the object
          obj_data = self.s3.get_object(Bucket=bucket_name, Key=key)
          # Read the object (image file) data into a stream
          image_stream = BytesIO(obj_data['Body'].read())
          # Load the image into a PIL.Image object and display it
          image = Image.open(image_stream)
          # append Image to Images
          images.append(image)
        return images

    # Preprocessing images
    def preprocess_images(self):
        image_paths = self.get_image_paths()
        images = self.get_images(image_paths)
        preprocessed_images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        image_features = self.model.encode_image(preprocessed_images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features_norm, image_paths


    # Saving image features
    def store_image_features(self, image_features, image_paths):
        target_path = f'{self.user_id}/processed/'
        # Serialize the tensor and write it to a byte stream
        byte_stream1 = BytesIO()
        torch.save(image_features, byte_stream1)
        byte_stream1.seek(0)  # Important: reset the position to the beginning of the stream
        self.s3.upload_fileobj(byte_stream1, bucket_name, target_path+"image_features.pt")

        # Store imagepaths
        byte_stream2 = BytesIO()
        byte_stream2.write('/n'.join(image_paths).encode('utf-8'))
        byte_stream2.seek(0)
        self.s3.upload_fileobj(byte_stream2, bucket_name, target_path+"image_paths.csv")