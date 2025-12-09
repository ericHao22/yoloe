from ultralytics import YOLOE
from ultralytics.models.yolo.segment.val_prompt_free import PromptFreeSegmentationValidator

model = YOLOE("pretrain/yoloe-v8l-seg.pt")
results = model.val(
    data="ultralytics/cfg/datasets/coco-cvpdl.yaml",
    batch=1,
    validator=PromptFreeSegmentationValidator,
    split='val',
    rect=False,
)

# Print specific metrics
print("Mean average precision:", f'{results.box.map:.2f}')
print("Mean average precision at IoU=0.50:", f'{results.box.map50:.2f}')
print("Mean average precision at IoU=0.75:", f'{results.box.map75:.2f}')
print("Mean precision:", f'{results.box.mp:.2f}')
print("Mean recall:", f'{results.box.mr:.2f}')
print("Mean F1 score:", f'{results.box.f1.mean():.2f}')