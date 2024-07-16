from ultralytics import YOLO

model = YOLO('models/best.pt')
result = model.predict('input_videos/A1606b0e6_0 (16).mp4',save = True)
print(result)
for box in result[0].boxes:
    print(box)