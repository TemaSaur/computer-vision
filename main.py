import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from collections import defaultdict as dd, Counter

PERSON_CLASSES = [0]
PLATED_VEHICLE_CLASSES = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]
UNPLATED_VEHICLE_CLASSES = [
    1,  # bicycle
    4,  # airplane
    6,  # train
    8,  # boat
]

VEHICLE_TYPE = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
    4: "airplane",
    6: "train",
    8: "boat",
}
ANIMAL_CLASSES = [
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
    64,  # mouse
]


class Recognizer:
	def __init__(self, video_source):
		self.model = YOLO('yolov8n.pt')
		self.cap = cv2.VideoCapture(video_source)

	def recognize(self):
		ret = True
		while ret:
			if (frame := self.get_frame()) is None:
				break

			result = self.find_objects(frame)
			plates = self.find_plates(frame, result.boxes) or []
			[plate.read() for plate in plates]
			self.visualize(result, plates)

			Logger.add_objects(result.boxes)
			Logger.add_plates(plates)

			ret = self.try_continue()

	def get_frame(self):
		ret, frame = self.cap.read()
		if not ret:
			return None
		return frame

	def find_objects(self, frame):
		classes = PERSON_CLASSES + PLATED_VEHICLE_CLASSES + UNPLATED_VEHICLE_CLASSES + ANIMAL_CLASSES
		return self.model.track(frame, persist=True, classes=classes)[0]

	def find_plates(self, frame, boxes):
		if boxes.id is None:
			return

		plates = []

		for box in Util.iter_boxes(boxes):
			if box.cls not in PLATED_VEHICLE_CLASSES:
				continue
			rectangle = Util.get_np_slice(box.xyxy)
			car_frame = frame[rectangle]

			plate = PlateReader(car_frame, box).find_plate()

			if plate is not None:
				plates.append(plate)
		return plates


	def visualize(self, result, plates):
		image = result.plot()
		image = Recognizer.add_plates(image, plates)
		cv2.imshow('video', image)

	def try_continue(self):
		return not (cv2.waitKey(1) & 0xFF == ord('q'))

	@staticmethod
	def add_plates(image, plates):
		for plate in plates:
			start = plate.car_plate_pos
			end = start[0] + plate.w, start[1] + plate.h
			image = cv2.rectangle(image, start, end, (255, 210, 200), 2)
			cv2.putText(image, plate.text, (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 210, 200), 2)
		return image

	def __enter__(self):
		return self

	def __exit__(self, *_):
		cv2.destroyAllWindows()
		Logger.write()


class PlateReader:
	haar = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

	def __init__(self, car_frame, car):
		self.car_frame = car_frame
		self.frame = None
		self.og_car_pos = car.xyxy[0], car.xyxy[1]
		self.car_id = car.id

	def find_plate(self):
		results = PlateReader.haar.detectMultiScale(self.car_frame,
		                                            minNeighbors=5)
		if len(results) == 0:
			return

		x, y, w, h = results[0]
		self.x = x + 5
		self.y = y + 5
		self.w = w - 5
		self.h = h - 5

		self.frame = self.car_frame[self.xyxy]
		return self

	def read(self):
		if self.frame.size == 0:
			return
		frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
		self.text = pytesseract.image_to_string(
		    frame,
			lang="eng",
			config="-c tessedit_char_whitelist=ABCEHKMOPTXY0123456789 --psm 10",
		)

	@property
	def xyxy(self):
		return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)

	@property
	def car_plate_pos(self):
		return self.og_car_pos[0] + self.x, self.og_car_pos[1] + self.y


class Logger:
	class Node:
		def __init__(self, id, cls):
			self.id = id
			self.cls = VEHICLE_TYPE[cls]
			self.plates = []

		def __repr__(self):
			return f'{self.id}: [{self.cls}] {" ".join(self.plates)}'

		def __str__(self):
			if self.cls in UNPLATED_VEHICLE_CLASSES:
				return f'id {self.id}: [{self.cls}]'

			plate = 'unknown plate'
			plates = Counter(self.plates).most_common(5)
			for p in plates:
				if 7 <= len(p[0]) <= 9:
					plate = p

			return f'id: {self.id}: [{self.cls}], plate: {plate}'

	vehicles = {}

	@staticmethod
	def add_objects(boxes):
		if boxes.id is None:
			return
		for box in Util.iter_boxes(boxes):
			if box.cls not in PLATED_VEHICLE_CLASSES + UNPLATED_VEHICLE_CLASSES:
				continue
			vehicle = Logger.vehicles.get(box.id)
			if vehicle is None:
				Logger.vehicles[box.id] = Logger.Node(box.id, box.cls)

	@staticmethod
	def add_plates(plates):
		for plate in plates:
			vehicle = Logger.vehicles[plate.car_id]
			text = plate.text.replace(' ', '').replace('\n', '')
			if text == '':
				continue
			vehicle.plates.append(text)

	@staticmethod
	def write():
		with open('log.txt', 'w') as f:
			for vehicle in Logger.vehicles.values():
				f.write(str(vehicle) + '\n')


class Box:
	def __init__(self, id, cls, xyxy):
		self.id = id
		self.cls = cls
		self.xyxy = xyxy


class Util:
	@staticmethod
	def iter_boxes(boxes):
		for id, cls, xyxy in zip(Util.get_ints(boxes.id),
		                         Util.get_ints(boxes.cls),
		                         Util.get_int_lists(boxes.xyxy)):
			yield Box(id, cls, xyxy)

	@staticmethod
	def get_ints(tensors):
		return tensors.int().tolist()

	@staticmethod
	def get_int_lists(xyxys):
		return map(lambda xyxy: Util.get_ints(xyxy), xyxys)

	@staticmethod
	def get_np_slice(xyxy):
		return slice(xyxy[1], xyxy[3]), slice(xyxy[0], xyxy[2])


if __name__ == '__main__':
	with Recognizer('./videos/msc.mp4') as recognizer:
		recognizer.recognize()
