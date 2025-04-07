import argparse
import train
import test
import eval
from datasets.dataset_trimble_slabs import TrimbleSlabs
from models import ctrbox_net
import decoder
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=1024, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=1024, help='Resized image width')
    parser.add_argument('--K', type=int, default=100, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_50.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='TrimbleSlabs', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='/CracksData/Dataset_4096_2048_0.5', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval, val}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    parser.add_argument('--output_dir', type=str, default='', help='Ouput directory for testing images')
    parser.add_argument('--save_path', type=str, default='', help='Output dir for pth files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'TrimbleSlabs': TrimbleSlabs}
    num_classes = {'TrimbleSlabs': 1}
    heads = {'hm': num_classes[args.dataset],
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

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio,
                                       down_ratio_kpts=down_ratio_kpts,
                                       kpts_radius=kpts_radius,
                                       neptune_project="arijus/OPEN3D-ML", 
                                       neptune_api="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMDZlMWNkYi0xZjA5LTQ0MjQtOGVkZS0wZGU3YjdmZTliNmYifQ==")

        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        os.makedirs(args.output_dir, exist_ok=True)
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio, down_ratio_kpts=down_ratio_kpts, output_dir=args.output_dir)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio, down_ratio_kpts=down_ratio_kpts,)