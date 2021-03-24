from flask import Flask, jsonify, request
import sys, os
import shutil
import cv2
import numpy as np
import traceback
import keras
import base64

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect

from src.keras_utils 		import load_model
from glob 					import glob
from src.utils 				import im2single
from src.keras_utils 		import load_model, detect_lp
from src.label 				import Shape, writeShapes

from src.label				import dknet_label_conversion
from src.utils 				import nms

from os.path 				import isfile
from src.drawing_utils		import draw_label, draw_losangle, write2img
from src.label 				import lread, Label, readShapes

from pdb import set_trace as pause

app = Flask(__name__)


@app.route('/',methods = [ 'POST', 'GET'])
def hello_world():
    if(request.method == 'GET'):

		return ' Heloo'
	
    img_base64 = request.json.get('img', None)
    if not img_base64:
        return jsonify('erro ao pegar a imagem base64')
    uuid = request.json.get('uuid', None)
    if not uuid:
        return jsonify('erro ao pegar o uuid')
    imgdata = base64.b64decode(img_base64)
    filename = 'images/'+uuid+'.jpg'
    output = 'images/output/'+uuid
    with open(filename, 'wb') as f:
        f.write(imgdata)
    vehicle_detection(filename, output)
    license_plate_detection(output , 'data/lp-detector/wpod-net_update1.h5')
    lp = license_plate_ocr(output)
    json = {"placa":lp}
    os.remove(filename)
    shutil.rmtree('images/output')
    return jsonify(json)



def vehicle_detection(filename, output_dir):
    try:
        vehicle_threshold = .5 
        vehicle_weights = 'data/vehicle-detector/yolo-voc.weights' 
        vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
        vehicle_dataset = 'data/vehicle-detector/voc.data' 

        vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
        vehicle_meta = dn.load_meta(vehicle_dataset)

        if not isdir(output_dir):
			makedirs(output_dir)
        
        print 'Searching for vehicles using YOLO...'

        print '\tScanning %s' % filename

        bname = basename(splitext(filename)[0])

        R,_ = detect(vehicle_net, vehicle_meta, filename ,thresh=vehicle_threshold)

        R = [r for r in R if r[0] in ['car','bus']]

        print '\t\t%d cars found' % len(R)

        if len(R): 

            Iorig = cv2.imread(filename)
            WH = np.array(Iorig.shape[1::-1],dtype=float)
            Lcars = []
            for i,r in enumerate(R):
                cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                tl = np.array([cx - w/2., cy - h/2.]) 
                br = np.array([cx + w/2., cy + h/2.])
                label = Label(0,tl,br)
                Icar = crop_region(Iorig,label)
                Lcars.append(label)
                cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)
            lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)
    except:
		traceback.print_exc()
		sys.exit(1)



def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

def license_plate_detection(input_dir, wpod_net_path):
    try:
		output_dir = input_dir
		lp_threshold = .5
		wpod_net = load_model(wpod_net_path)
		imgs_paths = glob('%s/*car.png' % input_dir)

		print 'Searching for license plates using WPOD-NET'

		for i,img_path in enumerate(imgs_paths):

			print '\t Processing %s' % img_path

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

			Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
		keras.backend.clear_session()
    except:
		traceback.print_exc()
		sys.exit(1)


def license_plate_ocr(input_dir):
    try:
	
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print 'Performing OCR...'

		for i,img_path in enumerate(imgs_paths):

			print '\tScanning %s' % img_path

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print '\t\tLP: %s' % lp_str

				return lp_str
			else:

				print 'No characters found'
                return ' '

    except:
		traceback.print_exc()
		sys.exit(1)



