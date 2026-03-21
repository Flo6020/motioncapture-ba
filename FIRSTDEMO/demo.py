from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
    pose2d_weights='td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
)

results = list(inferencer('VNpose.jpg', out_dir='output'))

print("fertig ")
