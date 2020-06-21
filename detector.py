import os
import cv2
import numpy as np
import pytesseract


class CharacterDetector(object):
    @staticmethod
    def process_image(license_plate_image, input_width, input_height):
        yolo_cfg = 'resources/character_detector.cfg'
        yolo_name = 'resources/character.names'
        yolo_weight = 'models/character_last.weights'
        scale = 0.00392
        yolo_input_size = (416, 416)
        with open(yolo_name, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet(yolo_weight, yolo_cfg)
        license_plate_image = cv2.resize(license_plate_image, (416, 416))
        # create input blob 0.002,0.00170,latest-0.0019
        blob = cv2.dnn.blobFromImage(license_plate_image, scale, yolo_input_size, (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)
        outs = net.forward(LicensePlateDetector.get_output_layers(net))
        class_ids = []
        cls_boxes = [[] for i in range(len(classes))]
        cls_confidences = [[] for i in range(len(classes))]
        cls_ids = [[] for i in range(len(classes))]
        conf_threshold_list = [0.3 for i in range(len(classes))]
        nms_class_thresh = [0.5 for i in range(len(classes))]
        Width = input_width
        Height = input_height
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                conf_threshold = conf_threshold_list[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    cls_ids[class_id].append(class_id)
                    cls_boxes[class_id].append([x, y, w, h])
                    cls_confidences[class_id].append(float(confidence))
        output_boxes = []
        final_classes = []
        for cls in range(len(cls_boxes)):
            nms_threshold = nms_class_thresh[cls]
            indices = cv2.dnn.NMSBoxes(cls_boxes[cls], cls_confidences[cls], conf_threshold_list[cls], nms_threshold)
            boxes = cls_boxes[cls]
            class_id = cls_ids[cls]
            for i in indices:
                i = i[0]
                box = boxes[i]
                x1 = int(box[0])
                y1 = int(box[1])
                w = int(box[2])
                h = int(box[3])
                x2 = x1 + w
                y2 = y1 + h
                final_classes.append(class_id[i])
                output_boxes.append([x1, y1, x2, y2])

        return output_boxes


class LicensePlateDetector(object):
    @staticmethod
    def extract_text(image_list):
        # Change this path to locally installed tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\VivekStorm\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
        output_number_plate_list = []
        for i, each_plate_image in enumerate(image_list):
            cv2.imwrite(str(i) + '_plate.jpg', each_plate_image)
            output_number = pytesseract.image_to_string(each_plate_image)
            output_number = ''.join(e for e in output_number if e.isalnum())
            output_number_plate_list.append(output_number)
        return output_number_plate_list

    @staticmethod
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    @staticmethod
    def draw_bounding_box(image, box, color=(0, 255, 255)):
        """
        Description : Draws bounding box of predicted coords are returns cropped image of prediction
        image : input image
        box : list of prediction coords
        """
        cropped_image_list = []
        draw_image = image.copy()
        for each_box in box:
            x1, y1, x2, y2 = each_box
            x1 = int(max(0, x1 - 3))
            y1 = int(max(0, y1 - 3))
            x2 = x2 - 2
            y2 = y2 - 2
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_list.append(cropped_image)
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 2)
        return draw_image, cropped_image_list

    @staticmethod
    def process_image(car_image, input_width, input_height):
        """
        Description : Given a image, this function will detect license plate and returns coordinates
        car_image : numpy image array
        input_width :  width of the car_image
        input_height: Height of the car_image
        """
        yolo_cfg = 'resources/numberplate_new.cfg'
        yolo_name = 'resources/numberplate.names'
        yolo_weight = 'models/spine_last_new.weights'
        scale = 0.00392
        yolo_input_size = (416, 416)
        with open(yolo_name, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet(yolo_weight, yolo_cfg)
        car_image = cv2.resize(car_image, (416, 416))
        # create input blob 0.002,0.00170,latest-0.0019
        blob = cv2.dnn.blobFromImage(car_image, scale, yolo_input_size, (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)
        outs = net.forward(LicensePlateDetector.get_output_layers(net))
        class_ids = []
        cls_boxes = [[]]  # number of classes
        cls_confidences = [[]]  # number of
        cls_ids = [[]]
        conf_threshold_list = [0.3]
        nms_class_thresh = [0.2]
        Width = input_width
        Height = input_height
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                conf_threshold = conf_threshold_list[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    cls_ids[class_id].append(class_id)
                    cls_boxes[class_id].append([x, y, w, h])
                    cls_confidences[class_id].append(float(confidence))
        output_boxes = []
        for cls in range(len(cls_boxes)):
            nms_threshold = nms_class_thresh[cls]
            indices = cv2.dnn.NMSBoxes(cls_boxes[cls], cls_confidences[cls], conf_threshold_list[cls], nms_threshold)
            boxes = cls_boxes[cls]
            for i in indices:
                i = i[0]
                box = boxes[i]
                x1 = int(box[0])
                y1 = int(box[1])
                w = int(box[2])
                h = int(box[3])
                x2 = x1 + w
                y2 = y1 + h
                output_boxes.append([x1, y1, x2, y2])
        return output_boxes


if __name__ == "__main__":
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path = 'test2'
    for each_image in os.listdir(test_path):
        current_image = os.path.join(test_path, each_image)
        car_image = cv2.imread(current_image)
        im_h, im_w = car_image.shape[0], car_image.shape[1]
        out_box = LicensePlateDetector.process_image(car_image, im_w, im_h)
        if out_box:
            o_img, license_plate_image_list = LicensePlateDetector.draw_bounding_box(car_image.copy(), out_box)
            output_image = os.path.join(output_path, each_image)
            cv2.imwrite(output_image, o_img)
            number_plate_list = LicensePlateDetector.extract_text(license_plate_image_list)
            print(number_plate_list)
    # character_test_path = 'character_test'
    # for each_char_img in os.listdir(character_test_path):
    #     current_image_path = os.path.join(character_test_path, each_char_img)
    #     current_image = cv2.imread(current_image_path)
    #     output_box = CharacterDetector.process_image(current_image, current_image.shape[1], current_image.shape[0])
    #     o_img,_ = LicensePlateDetector.draw_bounding_box(current_image.copy(),output_box)
    #     output_image = os.path.join(output_path,each_char_img)
    #     cv2.imwrite(output_image,o_img)
