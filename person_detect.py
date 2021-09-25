
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        detects = []
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords, width):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            xmin = int(coord[3] * width)
            xmax = int(coord[5] * width)
                
            for i, q in enumerate(self.queues):
                if xmin>q[0] and xmax<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.plugin = IECore()
        self.exec_net = self.plugin.load_network(self.model, self.device, num_requests=1)
        
    def predict(self, image):
        input_name = self.input_name
        input_img = self.preprocess_input(image)
        input_dict = {input_name: input_img}  
        # Start asynchronous inference for specified request.
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
        return outputs
    
    def draw_outputs(self, coords, image, init_w, init_h):
        detects = []
        for obj in coords[0][0]:
            # Draw bounding box for the detected object when it's probability 
            if obj[2] > self.threshold:
                xmin = int(obj[3] * init_w)
                ymin = int(obj[4] * init_h)
                xmax = int(obj[5] * init_w)
                ymax = int(obj[6] * init_h)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                detects.append(obj)
        return detects

    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''
        raise NotImplementedError

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        input_img=cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        # Change image from HWC to CHW
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape((n, c, h, w))
        return input_img


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            frameTest = queue.get_queues(frame)
            
            coords = pd.predict(frame)
            coords = pd.draw_outputs(coords, frame, initial_w, initial_h)
            
            num_people= queue.check_coords(coords, initial_w)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            image = frame
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)