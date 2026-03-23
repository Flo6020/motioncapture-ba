#Real-Time Multi-Person Pose Estimation --tiny
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='rtmpose-t_8xb256-420e_coco-256x192'
)

results = inferencer(
    'VNpose.jpg',
    show=False, #in meiner WSL-/Linux-Umgebung funktioniert das Qt/X11-Fenster-Backend nich richtig deswegen auf False
    out_dir='output'
)

for _ in results:
    pass

print("fertig")