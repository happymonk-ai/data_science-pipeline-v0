#Import Requiremnt for nats 
import asyncio
import nats
from io import BytesIO
from tkinter import Image
import numpy as np 
from PIL import Image
import cv2
import time 
import gc

#Import Requiremnt Yolov5
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from nanoid import generate

#PytorchVideo
from functools import partial

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import glob

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection # Another option is slowfast_r50_detection

from visualization import VideoVisualizer 

#Re_id
from itertools import count
# for everything else
import pickledb
import json
import face_recognition 

#Alpr
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json


device = 'cuda' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)
person_count =[]
vehicle_count = []
diff_detect =[]
diff_activity =[]
face_encoding_store = []
did_store = []
personDid=[]
TOLERANCE = 0.62
MODEL = 'cnn'
count_person = 0
activity_list = []
activity_list_box = []
license =[]
license_plate =[]
Device_id = []
frame_timestamp=[]
Geo_location = []
calc_timestamps = [0.0]
avg_Batchcount_person =[]
avg_Batchcount_vehicel =[]
ref_pixel = []
dist_list =[]
boundry_detected_person = []
boundry_detected_vehicle =[]
detect_count = []
track_person =[]
track_vehicle = []
vehicle_known_whitelist = ["KA01AB1234","KA22EN7880","KA02CD5678"]
vehicle_known_blacklist = ["KA02AB1004","KA02EN7880","KA03CD9876"]
face_did_encoding_store = dict()
batch_person_id = []
track_type = []

# model1 = keras.models.load_model("/home/wajoud/Datascience_pipline/Testing/license_plate.h5")
df = []
min_max_scaler = preprocessing.MinMaxScaler()
cnt=0
t=[]
c_cnt = []


# print('Loading known faces...')
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
track_type = []

#Alpr 
# def load_model(path):
#     try:
#         path = splitext(path)[0]
#         with open('%s.json' % path, 'r') as json_file:
#             model_json = json_file.read()
#         model2 = model_from_json(model_json, custom_objects={})
#         model2.load_weights('%s.h5' % path)
#         print("Loading model successfully...")
#         return model2
#     except Exception as e:
#         print(e)
        
# wpod_net_path = "wpod-net.json"
# wpod_net = load_model(wpod_net_path)

        
# #pickledb_whitelist   

# Move this to a function
# And Call the code in the main before the other code starts. 
# You can thread the part. 
db_whitelist = pickledb.load("Weights/known_whitelist.db", True)
list1 = list(db_whitelist.getall())

db_count_whitelist = 0
#This should be made as a function and called. its a repeat code 
for name in list1:    
        # Next we load every file of faces of known person
        re_image = db_whitelist.get(name)

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decodedArrays = json.loads(re_image)

        finalNumpyArray = np.asarray(decodedArrays["array"],dtype="uint8")
        
        # Load an image
        # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        image = finalNumpyArray
        ratio = np.amax(image) / 256        
        image = (image / ratio).astype('uint8')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        try : 
            encoding = face_recognition.face_encodings(image)[0]
        except IndexError as e  :
            print( "Error ", IndexError , e)
            continue

        # Append encodings and name
        known_whitelist_faces.append(encoding)
        known_whitelist_id.append(name)
        db_count_whitelist += 1
print(db_count_whitelist, "total whitelist person")


#Combine this with the above function and call it once. 
#pickledb_balcklist  
db_blacklist = pickledb.load("Weights/known_blacklist.db", True)
list1 = list(db_blacklist.getall())

#Combine this with the above function. 
db_count_blacklist = 0
for name in list1:    
        # Next we load every file of faces of known person
        re_image = db_blacklist.get(name)

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decodedArrays = json.loads(re_image)

        finalNumpyArray = np.asarray(decodedArrays["array"],dtype="uint8")
        
        # Load an image
        # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        image = finalNumpyArray
        ratio = np.amax(image) / 256        
        image = (image / ratio).astype('uint8')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        try : 
            encoding = face_recognition.face_encodings(image)[0]
        except IndexError as e  :
            print( "Error ", IndexError , e)
            continue

        # Append encodings and name
        known_blacklist_faces.append(encoding)
        known_blacklist_id.append(name)
        db_count_blacklist += 1
print(db_count_blacklist, "total blacklist person")

# Move this to function and call the function

def load_models():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    count_video = 0 
    return



async def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

async def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), ori_boxes

#Move the function name from Activity to acitivity_inference
async def Activity(source):
            # Create an id to label name mapping
            global count_video            
            label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('./Weights/ava_action_list.pbtxt')
            # Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
            video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)
            
            encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(source)
            
            time_stamp_range = range(1,25) # time stamps in video for which clip is sampled. 
            clip_duration = 1.0 # Duration of clip used for each inference step.
            gif_imgs = []
            
            for time_stamp in time_stamp_range:    
                print("Generating predictions for time stamp: {} sec".format(time_stamp))
                
                # Generate clip around the designated time stamps
                inp_imgs = encoded_vid.get_clip(
                    time_stamp - clip_duration/2.0, # start second
                    time_stamp + clip_duration/2.0  # end second
                )
                inp_imgs = inp_imgs['video']
                
                # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
                # We use the the middle image in each clip to generate the bounding boxes.
                inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
                inp_img = inp_img.permute(1,2,0)
                
                # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
                predicted_boxes = await get_person_bboxes(inp_img, predictor) 
                if len(predicted_boxes) == 0: 
                    print("Skipping clip no frames detected at time stamp: ", time_stamp)
                    continue
                    
                # Preprocess clip and bounding boxes for video action recognition.
                inputs, inp_boxes, _ = await ava_inference_transform(inp_imgs, predicted_boxes.numpy())
                # Prepend data sample id for each bounding box. 
                # For more details refere to the RoIAlign in Detectron2
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                
                # Generate actions predictions for the bounding boxes in the clip.
                # The model here takes in the pre-processed video clip and the detected bounding boxes.
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                preds = video_model(inputs, inp_boxes.to(device))

                preds= preds.to('cpu')
                # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
                preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
                
                # Plot predictions on the video and save for later visualization.
                inp_imgs = inp_imgs.permute(1,2,3,0)
                inp_imgs = inp_imgs/255.0
                out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
                gif_imgs += out_img_pred
                 

            try:
                height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]
                vide_save_path = './Nats_output/output_pytorch'+str(count_video)+'.mp4'
                video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))
            
                for image in gif_imgs:
                    img = (255*image).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video.write(img)
                video.release()

            except IndexError:
                print("No Activity")
                activity_list.append("No Activity")

            count_video += 1

@torch.no_grad()
async def detect(
        weights="./Weights/best.pt",  # model.pt path(s)
        source=ROOT,  # file/dir/URL/glob, 0 for webcam
        data="data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'Nats_output/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    person_count = []
    reference_length = 10.6 
    crossing_threshold = 0.1 
    ref_pixel_mean = 885.5414053925799 
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    #Alpr 
    
    # Match contours to license plate or character template
    def find_contours(dimensions, img,cnt) :

        # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]
        
        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs :
            #detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            
            #checking the dimensions of the contour to filter out the characters by contour's size
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

                char_copy = np.zeros((44,24))
                #extracting each character using the enclosing rectangle's coordinates.
                char = img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))
                
                # cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

    #           Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy) #List that stores the character's binary image (unsorted)
                
        #Return characters on ascending order with respect to the x-coordinate (most-left character first)
                
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = np.array(img_res_copy)

        return img_res

    # Find characters in the resulting images
    def segment_characters(image,cnt) :
        # Preprocess cropped license plate image
        img_lp = cv2.resize(image, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        dimensions = [LP_WIDTH/6,
                        LP_WIDTH/2,
                        LP_HEIGHT/10,
                        2*LP_HEIGHT/3]

        # Get contours within cropped license plate
        char_list = find_contours(dimensions, img_binary_lp,cnt)

        return char_list,len(char_list)


    def fix_dimension(img): 
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img
  
    def show_results(char):
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,c in enumerate(characters):
            dic[i] = c
        output = []
        for i,ch in enumerate(char): #iterating over the characters
            img_ = cv2.resize(ch, (28,28))
            img = fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = model1.predict(img)#predicting the class
            classes_y=np.argmax(y_,axis=1)
            character = dic[classes_y[0]] #
            output.append(character) #storing the result in a list
            
        plate_number = ''.join(str(v) for v in output)
        return plate_number
    
    def Lp_detect(file):
        try:
            Dmax=610 
            Dmin=258
            image_rgb = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
            pixels = image_rgb.astype('float32')
            pixels /= 255.0
            ratio = float(max(pixels.shape[:2])) / min(pixels.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)
            _ , LpImg, _, cor = detect_lp(wpod_net, pixels , bound_dim, lp_threshold=0.5)
            crop_img = np.array(255*LpImg[0], dtype = 'uint8')
            return crop_img
        except Exception as e:
            print("error" , e)
            pass

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    global vehicle_count , license
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == "person" :
                        person_count.append(f"{n}")
                    if names[int(c)] == "vehicle":
                        vehicle_count.append(f"{n}")
            
               
                
                for c in det[:,-1]:
                    global personDid , count_person ,license_plate
                    # if count == 10: 
                    if names[int(c)]=="person":
                        count_person += 1
                        if count_person>0:
                            np_bytes2 = BytesIO()
                            np.save(np_bytes2, im0, allow_pickle=True)
                            np_bytes2 = np_bytes2.getvalue()

                            image = im0 # if im0 does not work, try with im1
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            # print(MODEL, image ,"model ,image")
                            locations = face_recognition.face_locations(image, model=MODEL)
                            # print(locations,"locations 602")

                            encodings = face_recognition.face_encodings(image, locations)
                            
                            print(f', found {len(encodings)} face(s)\n')
                            
                            for face_encoding ,face_location in zip(encodings, locations):
                                    print(np.shape(known_whitelist_faces), "known_whitelist_faces", np.shape(face_encoding),"face_encoding")
                                    results_whitelist = face_recognition.compare_faces(known_whitelist_faces, face_encoding, TOLERANCE)
                                    print(results_whitelist, "611")
                                    if True in results_whitelist:
                                        did = '00'+ str(known_whitelist_id[results_whitelist.index(True)])
                                        print(did, "did 613")
                                        batch_person_id.append(did)
                                        track_type.append("00")
                                        if did in face_did_encoding_store.keys():
                                            face_did_encoding_store[did].append(face_encoding)
                                        else:
                                            face_did_encoding_store[did] = list(face_encoding)
                                    else:
                                        results_blacklist = face_recognition.compare_faces(known_blacklist_faces, face_encoding, TOLERANCE)
                                        print(results_blacklist,"621")
                                        if True in results_blacklist:
                                            did = '01'+ str(known_blacklist_id[results_blacklist.index(True)])
                                            print("did 623", did)
                                            batch_person_id.append(did)
                                            track_type.append("01")
                                            if did in face_did_encoding_store.keys():
                                                face_did_encoding_store[did].append(face_encoding)
                                            else:
                                                face_did_encoding_store[did] = list(face_encoding)
                                        else:
                                            if len(face_did_encoding_store) == 0:
                                                did = '10'+ str(generate(size =4 ))
                                                print(did, "did 642")
                                                track_type.append("10")
                                                batch_person_id.append(did)
                                                face_did_encoding_store[did] = list(face_encoding)
                                            else:
                                                # print(face_did_encoding_store,"face_did_encoding_store")
                                                for key, value in face_did_encoding_store.items():
                                                    print(key,"640")
                                                    if key.startswith('10'):
                                                        print(type(value),"type vlaue")
                                                        print(np.shape(np.transpose(np.array(value))), "value 642" ,np.shape(value) ,"value orginal",np.shape(face_encoding), "face_encoding")
                                                        results_unknown = face_recognition.compare_faces(np.transpose(np.array(value)), face_encoding, TOLERANCE)
                                                        print(results_unknown,"635")
                                                        if True in results_unknown:
                                                            key_list = list(key)
                                                            key_list[1] = '1'
                                                            key = str(key_list)
                                                            print(key, "did 637")
                                                            batch_person_id.append(key)
                                                            track_type.append("11")
                                                            face_did_encoding_store[key].append(face_encoding)
                                                        else:
                                                            did = '10'+ str(generate(size=4))
                                                            print(did, "did 642")
                                                            batch_person_id.append(did)
                                                            face_did_encoding_store[did] = list(face_encoding)
                                    print(batch_person_id, "batch_person_id")

                                
                                #     if True in results_store:
                                #         did = did_store[results_store.index(True)]
                                #         personDid.append(did)
                                #     elif True in results:
                                #         match = know_name[results.index(True)]
                                #         print("Match found: ", match)
                                #         did = generate(size=24)
                                #         alertLevel = 0
                                #         personDid.append(did)
                                #     else:
                                #         did = generate(size=24)
                                #         alertLevel = 1
                                #         personDid.append(did)
                                #     face_encoding_store.append(face_encoding)
                                #     did_store.append(did)
                                # # print("did store " , did_store)
                    
                        elif names[int(c)]=="vehicle":
                            print("line 581")
                            # print(im0.shape,"shape")
                            # crop = Lp_detect(im0)
                            # print(crop , "crop")
                            # char,k = segment_characters(crop,cnt)
                            # t=show_results(char)
                            # license_plate.append(str(t))
                        
                
                        
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #Boundry Detection 
                    if  names[int(torch.tensor(cls).tolist())] == "person":
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        a = np.shape(im0)
                        width = a[0]*xywh[2] 
                        height = a[1]*xywh[3]
                        reference_pixel = np.sqrt(width**2 + height**2)
                        distance = (ref_pixel_mean*reference_length)/reference_pixel
                        distance = abs(distance - reference_length)
                        if distance < crossing_threshold:
                            print("person crossing boundry")
                            boundry_detected_person.append(1)
                        else:
                            boundry_detected_person.append(0)
                    
                    if  names[int(torch.tensor(cls).tolist())] == "vehicle":
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        a = np.shape(im0)
                        width = a[0]*xywh[2] 
                        height = a[1]*xywh[3]
                        reference_pixel = np.sqrt(width**2 + height**2)
                        distance = (ref_pixel_mean*reference_length)/reference_pixel
                        distance = abs(distance - reference_length)
                        if distance < crossing_threshold:
                            print("person crossing boundry")
                            boundry_detected_vehicle.append(1)
                        else:
                            boundry_detected_vehicle.append(0)

                    # print(save_img, "Save Image ")
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # if len(did_store) != 0:
                        #     label = None if hide_labels else (names[c] , did_store[-1] if hide_conf else f'{names[c]} {conf:.2f}')
                        #     annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                                     
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    #people Count
    #Move this to a function, you can define another global variable for total People count and write the values down across all the cameras. 
    sum_count = 0
    for x in person_count:
        sum_count += int(x)
        if int(x) % 2 == 0:
            track_person.append(0)
        else:
            track_person.append(1)
    try :
        # avg = int(sum_count/len(person_count))
        avg_Batchcount_person.append(str(sum_count))
    except ZeroDivisionError:
        avg_Batchcount_person.append("0")
        print("No person found ")
        
    sum_count = 0
    for x in vehicle_count:
        sum_count += int(x)
        if int(x) % 2 == 0:
            track_vehicle.append(0)
        else:
            track_vehicle.append(1)
    try :
        # avg = int(sum_count/len(vehicle_count))
        avg_Batchcount_vehicel.append(str(sum_count))
    except ZeroDivisionError:
        avg_Batchcount_vehicel.append("0")
        print("No Vehicle found ")
        
    if len(person_count) > 0 or len(vehicle_count) > 0 :
        detect_count.append(1)
    else:
        detect_count.append(0)
     
    
    
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

        
#Move this to Logger function from AsyncIO        
async def error_cb(e):
    print("There was an Error:{e}", e)

async def BatchJson(source):
    global activity_list ,activity_list_box , person_count
    # We open the text file once it is created after calling the class in test2.py
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('./Weights/ava_action_list.pbtxt')
    # We open the text file once it is created after calling the class in test2.py
    file =  open('classes.txt', 'r')
    if file.mode=='r':
        contents= file.read()
    # print("Final Classes: ", contents)
    # Read activity labels from text file and store them in a list
    label = []
    # print('Content length: ', len(contents))
    for ind,item in enumerate(contents):
        if contents[ind]=='[' and contents[ind+1] == '[':
            continue
        if contents[ind]==']':
            if ind == len(contents)-1:
                break
            else:
                ind += 3
                continue
        if contents[ind]=='[' and contents[ind+1] != '[':
            ind += 1
            if ind>len(contents)-1:
                break
            label_each = []
            string = ''
            while contents[ind] != ']':
                if contents[ind]==',':
                    label_each.append(int(string))
                    string = ''
                    ind+=1
                    if ind>len(contents)-1:
                        break
                elif contents[ind]==' ':
                    ind+=1
                    if ind>len(contents)-1:
                        break
                else:
                    string += contents[ind]
                    ind += 1
                    if ind>len(contents)-1:
                        break
            if len(label_each)>0:
                label.append(label_each)
                label_each = []
    for item in label:
        activity_list_box = []
        for i in item:
            activity_list_box.append(label_map[i])
        activity_list.append(activity_list_box)
    return activity_list

            

async def main():
    global Device_id , frame_timestamp , geo_location ,license_plate,avg_Batchcount_person,avg_Batchcount_vehicel , detect_count , track_person , track_vehicle
    # nc = await nats.connect(servers=["nats://216.48.189.5:4222"] , reconnect_time_wait=5 ,allow_reconnect=True)
    # nc = await nats.connect(servers=["nats://216.48.181.154:4222"] , error_cb =error_cb ,reconnect_time_wait=2 ,allow_reconnect=True)
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , error_cb =error_cb ,reconnect_time_wait=2 ,allow_reconnect=True)
    # Create JetStream context.
    js = nc.jetstream()
    # psub = await js.pull_subscribe("Testing.video.frames1","psub", stream="Testing_stream1")
    psub = await js.pull_subscribe("stream.*.frame","psub", stream="device_stream")
    batch_size = 50
    while True:
        count = 0
        BatchId = generate(size=4)
        Device_id = []
        detect_count = []
        frame_timestamp = []
        license_plate =[]
        avg_Batchcount_person =[]
        avg_Batchcount_vehicel = []
        activity_list= []
        geo_location = []
        track_person = []
        track_vehicle = []
        boundry_detected_person = []
        gc.collect()
        torch.cuda.empty_cache()
        msgs = await psub.fetch(batch=batch_size , timeout = 50) 
        #For Python Nats Connection Start 
        # for msg in msgs:
        #     data = BytesIO(msg.data)
        #     data = np.load(data, allow_pickle =True)
        #     im = Image.fromarray(data)
        #     Device_id.append("1")
        #     frame_timestamp.append(str(generate(size=4)))
        #     Geo_location.append("latitude: 11.342423,longitude: 77.728165")
        #     im.save("Nats_output/output"+str(count)+".jpeg")
        #     await msg.ack()
        #     count+=1
        #For Python NATS Connection ends 
        #For Gstreamer Nats Connection start 
        for msg in msgs:
            data =(msg.data)
            data = data.decode('ISO-8859-1')
            parse = json.loads(data)
            device_id = parse['device_id']
            frame_code = parse['frame_bytes']
            timestamp = parse['timestamp']
            geo_location = parse['geo-location']
            frame_byte = frame_code.encode('ISO-8859-1')
            try :  
                arr = np.ndarray(
                    (720,
                    720),
                    buffer=np.array(frame_byte),
                    dtype=np.uint8)
                resized = cv2.resize(arr, (720 ,720))
                data1 = resized
                Device_id.append(device_id)
                # print("DEVICE_ID :", device_id)
                frame_timestamp.append(timestamp)
                # print("TIMESTAMP :", timestamp)
                Geo_location.append(geo_location)
                # print("Geo-location",geo_location)
                im = Image.fromarray(data1)
                im.save("Nats_output/output"+str(count)+".jpeg")
                # print("image saved")
            except TypeError as e:
                print(TypeError," gstreamer error >> ", e)
                continue        
            await msg.ack()
            count+=1
            #for Gstreamer Nats End 
            if count == batch_size:
                image_folder = 'Nats_output'
                video_name = './Nats_output/Nats_video.mp4'
                images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
                frame = cv2.imread(os.path.join(image_folder, images[0]))
                height, width, layers = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_name, fourcc , 1, (width,height))
                for image in images:
                    video.write(cv2.imread(os.path.join(image_folder, image)))
                video.release()
                time.sleep(2)
                start = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                await detect(source=video_name)
                diff_detect.append(time.time()-start)
                start = time.time()
                await Activity(source=video_name)
                gc.collect()
                torch.cuda.empty_cache()
                diff_activity.append(time.time()-start)
                activity_list = await BatchJson(source="classes.txt")
                # for item in batch_person_id:
                #     print(item, "item 937")

                #Move the JSON construct to the a 
                metapeople ={
                    "type":str(track_type),
                    "track":str(track_person),
                    "id":batch_person_id,
                    "activity":{"activities":activity_list , "boundaryCrossing":boundry_detected_person}  
                    }
    
                metaVehicle = {
                                "type":str(track_type),
                                "track":str(track_vehicle),
                                "id":license_plate,
                                "activity":{"boundaryCrossing":boundry_detected_vehicle}
                }
                metaObj = {
                            "people":metapeople,
                            "vehicle":metaVehicle
                        }
                
                metaBatch = {
                    "Detect": str(detect_count),
                    "Count": {"people_count":str(avg_Batchcount_person),
                             "vehicle_count":str(avg_Batchcount_vehicel)} ,
                            "Object":metaObj
                }
                
                primary = { "deviceid":str(Device_id[-1]),
                            "batchid":str(BatchId[-1]), 
                            "timestamp":str(frame_timestamp[-1]), 
                            "geo":str(Geo_location[-1]),
                            "metaData": metaBatch}
                print(primary)
                JSONEncoder = json.dumps(primary)
                json_encoded = JSONEncoder.encode()
                subjectactivity = "model.activity"
                await nc.publish(subjectactivity, json_encoded)
                print("Activity is getting published")
                
# Do all the preloading here. 
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        #use Logger function to log the value
        print("error ", e)
        #Check for memory error, if memory error, restart the main loop. 
        #use Logger function to log the print statement over here. 
        #Also print the device information with the memory information.
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
    
    
    
"""
Json Object For a Batch Video 

JsonObjectBatch= {ID , TimeStamp , {Data} } 
Data = {
    "person" : [ Device Id , [Re-Id] , [Frame TimeStamp] , [Lat , Lon], [Person_count] ,[Activity] ]
    "car":[ Device ID, [Lp Number] , [Frame TimeStamp] , [Lat , lon] ]
}  
Activity = [ "walking" , "standing" , "riding a bike" , "talking", "running", "climbing ladder"]

"""

"""
metapeople ={
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"face_id",
                    "activity":{"activities":activity_list , "boundaryCrossing":boundary}  
                    }
    
    metaVehicel = {
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"license_plate",
                    "activity":{"boundaryCrossing":boundary}
    }
    metaObj = {
                 "people":metapeople,
                 "vehicle":metaVehicel
               }
    
    metaBatch = {
        "Detect": "0: detection NO, 1: detection YES",
        "Count": {"people_count":str(avg_Batchcount),
                  "vehicle_count":str(avg_Batchcount)} ,
        "Object":metaObj
        
    }
    
    primary = { "deviceid":str(Device_id),
                "batchid":str(BatchId), 
                "timestamp":str(frame_timestamp), 
                "geo":str(Geo_location),
                "metaData": metaBatch}
    print(primary)
    
"""






