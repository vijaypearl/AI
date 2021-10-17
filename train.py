#To install MTCNN package use this url "https://pypi.org/project/mtcnn/"
#!pip install mtcnn
from mtcnn.mtcnn import MTCNN
#!pip install numpy
import numpy
#!pip install sklearn
from sklearn import neighbors
#!pip install face_recognition
import face_recognition
#!pip install opencv-python
import cv2
import math
import pickle
import warnings
import boto3
import io
warnings.filterwarnings("ignore")


# UTILS
access_key_id      = "AKIA4ORBUL5GXGTOFDOV"
secret_access_key  = "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"

def get_s3_paths_from_bucket(s3_bucket_name, org_id):
    s3_client = boto3.client('s3',
                         aws_access_key_id= access_key_id,
                         aws_secret_access_key=secret_access_key)

    response =  s3_client.list_objects_v2(Bucket=s3_bucket_name)
    
    keys =[]
    for obj in response['Contents']:
            keys.append(obj['Key'])
      
    people_dict ={}      
    for key in keys:
        if key.startswith(org_id):
            if key.endswith(".jpg") or key.endswith(".JPG") or key.endswith(".png") or key.endswith(".PNG"):
                #print(key)
                key_split_up   = key.split("/")
                unique_face_id = key_split_up[1]
                if unique_face_id in people_dict.keys():
                    people_dict[unique_face_id].append(key)
                else:
                    people_dict[unique_face_id] = [key]
    return people_dict


def get_image_from_s3(s3_bucket_name, s3_image_key):
    s3 = boto3.resource('s3',
                    aws_access_key_id= access_key_id,
                    aws_secret_access_key= secret_access_key)

    bucket = s3.Bucket(s3_bucket_name)
    image_obj = bucket.Object(s3_image_key)
    
    file_stream = io.BytesIO()
    image_obj.download_fileobj(file_stream)
    
    response = image_obj.get()
    image_stream = response['Body']
    
    return image_stream


def upload_model_to_s3(knn_clf, org_id, s3_bucket_name,  model_save_path):
    model_as_pickle = pickle.dumps(knn_clf)
        
    s3_resource = boto3.resource('s3',
                    aws_access_key_id= access_key_id,
                    aws_secret_access_key= secret_access_key)
    
    hello = s3_resource.Object(s3_bucket_name, org_id+ "/"+ model_save_path).put(Body= model_as_pickle)
    return hello

# Detection of FACE using MTCNN 
def face_detection_with_mtcnn(image, detector):
    
    """
    Takes Image location as input
    Returns the detection points of faces as output
    """
    pixels = numpy.array(image)

    # detect faces in the image
    faces = detector.detect_faces(pixels)
    
    # generating face_bounding_boxes
    face_bounding_boxes_mtcnn =[]
    for result in faces:
            # get coordinates
            x, y, width, height = result['box']
            xminn = x
            yminn = y
            xmaxx = x + width
            ymaxx = y + height
            face_bounding_boxes_mtcnn.append((yminn, xmaxx, ymaxx,  xminn))
    return face_bounding_boxes_mtcnn

# Training the recognition with below Util
def train(s3_bucket_name, org_id, model_save_path=None, n_neighbors=None, knn_algo='auto'):
    
    """
    Identifies the Face, Face embeddings from the given image
    Trains a k-nearest neighbors classifier for face recognition.
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    detector = MTCNN()
    
    
    people_dictionary = get_s3_paths_from_bucket(s3_bucket_name, org_id)
    
    for class_id, all_s3_image_keys in people_dictionary.items():
        for s3_image_key in all_s3_image_keys:
            
            image_stream = get_image_from_s3(s3_bucket_name, s3_image_key)
            
            image = face_recognition.load_image_file(image_stream)
            face_bounding_boxes = face_detection_with_mtcnn(image, detector)
            #image = face_recognition.load_image_file(img_path)
            #face_bounding_boxes = face_recognition.face_locations(image,number_of_times_to_upsample=1, model="cnn")

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                pass
            else:
                width = face_bounding_boxes[0][1] - face_bounding_boxes[0][3]
                height = face_bounding_boxes[0][2] - face_bounding_boxes[0][0]
                if width < 100 and height < 100:
                    if face_bounding_boxes[0][0]-20 >= 0 and face_bounding_boxes[0][3] - 20 >= 0:
                        face_image = image[face_bounding_boxes[0][0]-20:face_bounding_boxes[0][2] + 20, face_bounding_boxes[0][3] - 20:face_bounding_boxes[0][1] + 20] 
                    elif face_bounding_boxes[0][0]-20 < 0 and face_bounding_boxes[0][3] - 20 >= 0:
                        face_image = image[0:face_bounding_boxes[0][2] + 20, face_bounding_boxes[0][3] - 20:face_bounding_boxes[0][1] + 20]
                    elif face_bounding_boxes[0][0]-20 >= 0 and face_bounding_boxes[0][3] - 20 < 0:
                        face_image = image[face_bounding_boxes[0][0]-20:face_bounding_boxes[0][2] + 20, 0:face_bounding_boxes[0][1] + 20]
                    else:
                        face_image = image[0:face_bounding_boxes[0][2] + 20, 0:face_bounding_boxes[0][1] + 20]
                    face_image = cv2.resize(face_image, (face_image.shape[1]*3,face_image.shape[0]*3), interpolation = cv2.INTER_CUBIC)
                    face_bounding_boxes_2 = face_detection_with_mtcnn(face_image, detector)
                    X.append(face_recognition.face_encodings(face_image, known_face_locations=face_bounding_boxes_2, num_jitters=1, model='large')[0])
                else:
                    face_image = image#[face_bounding_boxes[0][0]-20:face_bounding_boxes[0][2] + 20, face_bounding_boxes[0][3] - 20:face_bounding_boxes[0][1] + 20] 
                    X.append(face_recognition.face_encodings(face_image, known_face_locations=face_bounding_boxes, num_jitters=1, model='large')[0])

                # Add face encoding for current image to the training set
                y.append(class_id)
    
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))


    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance',metric='euclidean')
    knn_clf.fit(X, y)
    
    # Save the trained KNN classifier
    if model_save_path is not None:
        try:
                upload_model_to_s3(knn_clf, org_id, s3_bucket_name, model_save_path)
                return True
        except:
                return False
    else:
        return False


# STEP 1: Train the KNN classifier and save it to disk
# Once the model is trained and saved, you can skip this step next time.
    
# Train the Face Model 
# Input parameters "s3 Bucket Name" where images are stored, "Organization id", ""
        

# status = train(s3_bucket_name= "irfacemodels", org_id= "4" , model_save_path= "trained_model_new.clf", n_neighbors=2)