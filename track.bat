@echo off
conda activate py38_byte
python tools/track_custom_kw.py -f exps/example/mot/yolox_x_custom_kw.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse

echo python tools/track_custom_kw.py -f exps/example/mot/yolox_x_custom_kw.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python tools/demo_custom_kw.py -f exps/example/mot/yolox_x_custom_kw.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse demo
python tools/demo_custom_kw.py -f exps/example/mot/yolox_x_custom_kw.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result image 

python tools/demo_custom_kw.py -f exps/example/mot/yolox_x_custom_kw.py -c pretrained/bytetrack_x_mot17.pth.tar --save_result image 