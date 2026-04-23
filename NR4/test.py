from mmdet.apis import DetInferencer

models = DetInferencer.list_models('mmdet')
for m in models:
    if 'rtmdet' in m.lower():
        print(m)