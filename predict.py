from PIL import Image, ImageDraw
import numpy
from sklearn import neighbors
import cv2
from mtcnn.mtcnn import MTCNN
import face_recognition
import pickle
import os
import io
import boto3
import warnings
warnings.filterwarnings("ignore")


# UTILS FOR PREDICTION
detector = MTCNN()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
access_key_id      = "AKIA4ORBUL5GXGTOFDOV"
secret_access_key  = "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"

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

def get_knn_clf(s3_bucket_name, org_id, model_name):    
    s3 = boto3.resource('s3',
                    aws_access_key_id= access_key_id,
                    aws_secret_access_key= secret_access_key)

    with io.BytesIO() as data:
        s3.Bucket(s3_bucket_name).download_fileobj(org_id+"/"+model_name, data)
        data.seek(0)    # move back to the beginning after writing
        classifier = pickle.load(data)
    
    return classifier

def show_prediction_labels_on_image(img_path, output_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom + text_height +2), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom + 2), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
    pil_image.save(output_path)


def predict(s3_model_bucket, s3_image_key, org_id, model_name, s3_image_bucket, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """    
    # distance_threshold = 0.6
    # To avoid false matches, use lower value
    # To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
    # 0.5-0.6 works well

    # Load a trained KNN model (if one was passed in)            
    knn_clf = get_knn_clf(s3_model_bucket, org_id,  model_name)

    # Load image file and find face locations
    image_stream = get_image_from_s3(s3_image_bucket, s3_image_key)
    X_img = face_recognition.load_image_file(image_stream)
    X_face_locations = face_detection_with_mtcnn(X_img, detector)
    faces_encodings = []
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    for face in X_face_locations:
        width = face[1] - face[3]
        height = face[2] - face[0]
        #print(width, height)
        if width < 100 and height < 100:
            if face[0]-20 >= 0 and face[3] - 20 >= 0:
                    face_image = X_img[face[0]-20:face[2] + 20, face[0][3] - 20:face[0][1] + 20] 
            elif face[0]-20 < 0 and face[3] - 20 >= 0:
                face_image = X_img[0:face[0][2] + 20, face[0][3] - 20:face[0][1] + 20]
            elif face[0][0]-20 >= 0 and face[0][3] - 20 < 0:
                face_image = X_img[face[0][0]-20:face[0][2] + 20, 0:face[0][1] + 20]
            else:
                face_image = X_img[0:face[0][2] + 20, 0:face[0][1] + 20]
            face_image = cv2.resize(face_image, (face_image.shape[1]*3,face_image.shape[0]*3), interpolation = cv2.INTER_CUBIC)
            face_bounding_boxes_2 = face_detection_with_mtcnn(face_image, detector)
            faces_encodings.append(face_recognition.face_encodings(face_image, known_face_locations=face_bounding_boxes_2, num_jitters=1, model='large')[0])
        else:
            face_image = X_img
            faces_encodings.append(face_recognition.face_encodings(face_image, known_face_locations=[face], num_jitters=1, model='large')[0])

    # Find encodings for faces in the test iamge
    #faces_encodings = face_recognition.face_encodings(X_img, known_face_locations = X_face_locations, num_jitters = 1)
    #print(faces_encodings)
    #print("Face Locations ", end="-")
    #print(X_face_locations)
    #print(X_img)
    
    
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings)
    #print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


# PREDICTION STARTS FROM HERE


# Find all people in the image using a trained classifier model
# Note: You can pass in either a classifier file name or a classifier model instance
#predictions = predict(full_file_path, model_path="trained_knn_model.clf")

def get_face_predictions(s3_model_bucket, s3_image_key, org_id, model_name, s3_image_bucket):
    predictions = predict(s3_model_bucket, s3_image_key, org_id, model_name, s3_image_bucket)

    output = []
    for name, (top, right, bottom, left) in predictions:
        #print("- Found {} at ({}, {})".format(name, left, top))
        rec_status = ""
        if name == "unknown":
            rec_status = "False"
        else:
            rec_status = "True"
            
        output.append([name, (top, right, bottom, left), rec_status])
    
    if len(output) > 0:
        return output
    else:
        return [["","","False"]]

# Usage
# To Predict the image provide "s3 Bucket name", "s3_image_key", "organization_id" and "model_name"
# s3_model_bucket = "irfacemodels"
# s3_image_bucket = "irfacebucket"
# s3_image_key    = "Hayner1.jpg"
# org_id          = "3"
# model_name      = "trained_knn_model_3.clf"

# print(get_face_predictions(s3_model_bucket = s3_model_bucket, s3_image_key = s3_image_key, org_id =org_id, model_name = model_name, s3_image_bucket = s3_image_bucket))


# Display results overlaid on an image
# show_prediction_labels_on_image(args.Input_Image_path ,args.Output_Image_path , predictions)

