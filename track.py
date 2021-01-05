import sys
sys.path.insert(0, './yolov5_deepSORT/yolov5')
from yolov5_deepSORT.yolov5.utils.datasets import LoadImages, LoadStreams, LoadRealsense
from yolov5_deepSORT.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_deepSORT.yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5_deepSORT.deepsort.utils.parser import get_config
from yolov5_deepSORT.deepsort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math
import multiprocessing as mp
import pyrealsense2.pyrealsense2 as rs
#Pytorch-openpose
# sys.path.append(os.path.dirname(os.path.abspath()))
from pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
from pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
from pose_estimation.modules.load_state import load_state
from pose_estimation.modules.pose import Pose, track_poses
from pose_estimation.val import normalize, pad_width
from DBConn import DB_Connection 
import timeit

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) # bounding box
        # cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1) #left up rectangle box 
        # cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 255], 2) # track id 

        # put trackining id when detected object is person
        # cv2.putText(img, 'id: {} '.format(label),(x1, y1 + t_size[0] + 4), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),1)

    return img


def detect(opt, net, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, height_size, cpu, track, smooth = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.height_size, opt.cpu, opt.track, opt.smooth
    
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # identify different devices
    ctx = rs.context()
    devices = ctx.query_devices()
    # print(devices[0])
    realsense = devices[0]

    # Openpose
    net = net.cuda()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)

    if realsense:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRealsense()

    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names(classname) and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # print('name', names) # list type
    

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # print('img shape-------00000', img.shape)
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    count_p = 0
    people = {}
    count_o = 0
    objects = {}
    customer_id = 0
    customer_name = ''
    product_id = 0
    product_name = ''

    db = DB_Connection()
    frame_num=0
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        start_t = timeit.default_timer()
        frame_num+=1
        #convert 
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # print ('img shape3333', img.shape) #384 640
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) # classes=opt.classes
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
        # for i, det in enumerate(pred):  # detections per image
            if webcam or realsense:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                bbox_xywh = []
                confs = []
                classes = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clsss = torch.Tensor(classes)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, clsss, im0)
                # print('outputsooooooooooooooooooooooo',outputs)

                # detections = [Detection(bbox_tlwh[i], conf, features[i], classes[i]) for i, conf in enumerate(
                #     confidences) if conf > self.min_confidence]
                # entity = [ if ]

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4:5]
                    class_id = outputs[:,-1]
                    count_P = np.count_nonzero(class_id == 0) # 사람만 카운트
                    count_O = np.count_nonzero(class_id) # 사람 아닌것만 카운트

                    # boudning box (detected object)
                    draw_boxes(im0, bbox_xyxy, identities)
                    # object center point
                    cv2.circle(im0,(bbox_xyxy[0][0] + int((bbox_xyxy[0][2] - bbox_xyxy[0][0])/2),bbox_xyxy[0][1] + int((bbox_xyxy[0][3] - bbox_xyxy[0][1])/2)),10, (0, 0, 255), -1)  # test
                    
            else:
                # 마지막 물체나 사람이 초기화 될시 그정보를 어떻게 이용할지!
                deepsort.increment_ages()
                count_p = 0
                people = {}
                count_o = 0
                objects = {}

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, 1 / t2 - t1))

#################################### openpose ############################################
        #im0 = im0.tolist()  => if webcam
        im0 = np.asarray(im0) 
        orig_img = im0.copy() 
        heatmaps, pafs, scale, pad = infer_fast(net, im0s[0], height_size, stride, upsample_ratio, cpu) #720 1280

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        ###### wrist ######
        current_wrists = []
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])

######################### comparing btw pose id & detected id  ##############################
            if det is not None and len(det):
                for o in outputs:
                    xyxy = o[:4]
                    class_id = o[-1]
                    track_id = o[4:5] ## added ##
                    if class_id == 0: # 사람일때 모든 사람의 중심 좌표와 맞나 검사
                        x1 = int(pose.bbox[0]) + int(pose.bbox[2]/2)
                        y1 = int(pose.bbox[1]) + int(pose.bbox[3]/2)
                        x2 = xyxy[0] + int((xyxy[2] - xyxy[0]) / 2)
                        y2 = xyxy[1] + int((xyxy[3] - xyxy[1]) / 2)
                        if cal_dist(x1, y1, x2, y2) < 70: # 두개의 객체가 사람일시 좌표값으로 동일한 객체인지 체크
                            current_poses.append(pose)
                            ### added ###
                            wrists = pose.draw(im0s[0])
                            current_wrists.append([track_id, wrists])
        # print('current_wrists',current_wrists)


########################### 카트에 물건을 담을때 쓰일 모든 기능들 ###########################
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        # for pose in current_poses:
        #     wrists = pose.draw(im0s[0])
     
            if det is not None and len(det):
                if len(outputs) > 0:
                    class_id = outputs[:,-1]
                    count_P = np.count_nonzero(class_id == 0) # 사람만 카운트
                    count_O = np.count_nonzero(class_id) # 사람 아닌것만 카운트
                    print('사람숫자:  ', count_p)
                    print('물건숫자:  ', count_O)
             
                    ########################### 사람과 물건을 딕셔너리에 삽입 ##############################
                    ######## 사람만있는 어레이를 만듬
                    p_list = []
                    people_copy = {}
                    [p_list.append(i) for i, outp in enumerate(outputs) if outp[-1] != 0]
                    p_outputs = np.delete(outputs, p_list, axis=0)

                    ######## 사람의 수가 증가시
                    if count_P > count_p:
                        #get checked in user id 
                        customer_id = db.select_user()
                        customer_name = db.get_username(customer_id)
                        if count_p == 0:
                            for p_output, wrist in zip(p_outputs, current_wrists): #for 문하나 더써서 넣기
                                people.setdefault(int(p_output[4:5]), [customer_id,])
                            count_p = count_P
                        else:
                            for p in people: # 새로운 사람들을 딕셔너리에 카피
                                for outp in p_outputs:
                                    if p != outp[4:5]:
                                        people_copy.setdefault(int(outp[4:5]))
                            #카피된 딕셔너리를 사람들에 넣기
                            for p in people_copy:
                                people.setdefault(p,customer_id)
                            count_p = count_P
                    elif count_P < count_p: # 사람수 감소시
                        k_list = p_outputs[:, 4:5]
                        for outp in people:
                            if outp not in k_list:
                                people_copy.setdefault(outp)
                        # 카피된 딕셔너리를 사람들에서 삭제
                        for o in list(people_copy):
                            del people[o]
                        count_p = count_P

                    ## 물건만 있는 어레이 만들기
                    o_list = []
                    objects_copy = {}
                    [o_list.append(i) for i, outp in enumerate(outputs) if outp[-1] == 0]
                    o_outputs = np.delete(outputs, o_list, axis=0)

                    ##### 물건수 증가시
                    if count_O > count_o:
                        #get product name
                        # product_id = db.select_product()
                        # product_name = db.get_productname(product_id)
                        # 물건이 하나도 없다가 카메라에 잡힐경우 모두 집어 넣기
                        if count_o == 0: 
                            for o in o_outputs:
                                objects.setdefault(int(o[4:5]),o[-1]) 
                                # insert customerId with DB
                                # insert_cart(customer_id,product_id,count_o)
                                print('count o type', type(count_o))
                            count_o = count_O
                        # 물건이 증가
                        else: 
                            for o in objects:
                                for outp in o_outputs:
                                    if o != outp[4:5]:
                                        objects_copy.setdefault(int(outp[4:5]), outp[-1])
                            for k, v in objects_copy.items():
                                objects.setdefault(k, v)
                            count_o = count_O
                    ###### 물건수 감소시
                    elif count_O < count_o: 
                        k_list = o_outputs[:,4:5]
                        for outp in objects:
                            if outp not in k_list:
                                objects_copy.setdefault(outp)
                                # db.remove_item()
                        # 카피된 딕셔너리를 사람들에서 삭제
                        for o in list(objects_copy):
                            del objects[o]
                        count_o = count_O
                    print('people people people ', people)
                    print('objects objects objects ', objects)
                else:
                    pass
            
                # cv2.putText(im0, 'person count: {}'.format(count_p), (480,700), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),2)
                # cv2.putText(im0, 'object count: {}'.format(count_O), (480,750), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),2)

            img = cv2.addWeighted(orig_img, 0.6, im0s[0], 0.4, 0) #720 1280
            idx = 0
            for pose in current_poses:
                #pose bounding box
                # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),(pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (255, 255, 0),2)
                cv2.circle(img, (int(pose.bbox[0]) + int(pose.bbox[2]/2), int(pose.bbox[1]) + int(pose.bbox[3]/2)), 10, (0, 255, 0), -1) # test
                #track id list 
                if track:
                    pose.id = customer_id
                    print('pose.id',pose.id)
                    # tracked person id (left top)
                    # cv2.putText(img, 'person id: {} is being tracked '.format(pose.id), (5,35+idx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),2)
                    cv2.putText(img, 'customer {}: {} is being tracked '.format(pose.id, customer_name), (5,35+idx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),2)
                    idx+=20
                #person id (above bbx)
                # cv2.putText(img, 'id: {} '.format(pose.id),(bbox_xyxy[0][0], bbox_xyxy[0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, 'id: {} '.format(pose.id),(bbox_xyxy[0][0], bbox_xyxy[0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

################# result #################
            terminate_t = timeit.default_timer()
            result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Grab_N_Go', result)
            FPS=int(1./(terminate_t-start_t))
            print('FPS',FPS)

    print('Done. (%.3fs)' % (time.time() - t0))


#######################################################################################################################

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    # print('img ============', img.shape)
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def cal_dist(x1,y1,x2,y2):
    dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5_deepSORT/yolov5/weights/yolov5x.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='realsense', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    # parser.add_argument('--classes', type=str,
    #                     default="/home/erin/Documents/project/ContactFree_Store_System/yolov5/data/coco.yaml", help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="yolov5_deepSORT/deepsort/configs/deep_sort.yaml")
    # Pytorch-openpose
    parser.add_argument('--checkpoint-path', type=str, default='pose_estimation/checkpoint/checkpoint_iter_370000.pth', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    #openpose
    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    # print(args)
    with torch.no_grad():
        detect(args, net)
