import torch
import torch.nn as nn
from models import ctrbox_net
from datasets.dataset_trimble_slabs import TrimbleSlabs


dataset = {'TrimbleSlabs': TrimbleSlabs}
num_classes = {'TrimbleSlabs': 1}
heads = {'hm': num_classes['TrimbleSlabs'],
            'wh': 10,
            'reg': 2,
            'cls_theta': 1,
            'hm_kpts': 1,
            'reg_kpts': 2
            }
down_ratio = 8
down_ratio_kpts = 8
kpts_radius = 3
model = ctrbox_net.CTRBOX(heads=heads,
                            pretrained=True,
                            down_ratio=down_ratio,
                            down_ratio_kpts=down_ratio_kpts,
                            final_kernel=1,
                            head_conv=256)


checkpoint = torch.load('/mnt/c/src/BBAVectors-Oriented-Object-Detection/CracksData/OP-1277_weights_TrimbleSlabs/model_38.pth', map_location=lambda storage, loc: storage)
state_dict_ = checkpoint['model_state_dict']
model.load_state_dict(state_dict_, strict=True)
model.eval()

original_forward = model.forward
model.forward = model.forward_full_onnx

dummy_input = torch.randn(1, 3, 1024, 1024)

output_names = ['hm', 'wh', 'reg', 'cls_theta', 'hm_kpts', 'reg_kpts'] # Get head names ['cls_theta', 'hm', 'hm_kpts', 'reg', 'reg_kpts', 'wh']
output_names.append("edge_probs")
print(f"Exporting with output names: {output_names}")

dynamic_axes = {'input': {0: 'batch_size'}} # Dynamic batch for input
for name in output_names:
    # Assuming all outputs have batch size on the first dimension
    # Adjust if some outputs have different dimension structures
    dynamic_axes[name] = {0: 'batch_size'}

dummy_input_tuple = (dummy_input,)

try:
    torch.onnx.export(
        model,
        dummy_input_tuple, # Pass inputs as a tuple
        "slabs_detection_full.onnx", # Use a new name to avoid confusion
        export_params=True,
        opset_version=11, # Consider 12+ if available/needed
        do_constant_folding=True,
        input_names=['input'], # Match tuple elements
        output_names=output_names, # Use the correct list of names
        dynamic_axes=dynamic_axes # Use the correct dict for all I/O
    )
    print("ONNX export successful!")

except Exception as e:
    print(f"ONNX export failed: {e}")

finally:
    # Restore the original forward method if needed for further use
    model.forward = original_forward
