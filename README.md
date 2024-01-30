# retico-dino
A ReTiCo module for DINO. See below for more information.

https://github.com/facebookresearch/dinov2

### Installation and requirements

### Example
```python
import sys, os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

prefix = '/path/to/prefix'
sys.path.append(prefix+'retico-core')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-sam')
sys.path.append(prefix+'retico-dino')

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule 
from retico_dino.dino import Dinov2ObjectFeatures
from retico_vision.vision import ExtractObjectsModule
from retico_sam.sam import SAMModule

path_var = 'sam_vit_h_4b8939.pth'

webcam = WebcamModule()
sam = SAMModule(model='h', path_to_chkpnt=path_var, use_bbox=True)  
extractor = ExtractObjectsModule(num_obj_to_display=1)  
feats = Dinov2ObjectFeatures(show=False, top_objects=1)
debug = DebugModule()  

webcam.subscribe(sam)  
sam.subscribe(extractor)  
extractor.subscribe(feats)    
feats.subscribe(debug)

webcam.run()  
sam.run()  
extractor.run()  
feats.run()
debug.run()  

print("Network is running")
input()

webcam.stop()  
sam.stop()  
extractor.stop()   
debug.stop()  
```

Citation
```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```