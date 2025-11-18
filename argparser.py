import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='screw_bag',
                        help='category')
    parser.add_argument('-o', '--output_dir', default='./results/')
    parser.add_argument('-w', '--weights', default='models/teacher_medium.pth')
    parser.add_argument('-i', '--imagenet_train_path', default='./data/imagenet/train',)
    parser.add_argument('--mvtec_loco_path', default='./data/mvtec_loco'),
    parser.add_argument('--mvtec_loco_seg_path', default='./data/mvtec_loco_composition_maps/',)
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_relation_branch', action='store_true', help='Enable relation branch during inference')
    parser.add_argument('--relation_model_path', default='none', help='Path to a trained relation model checkpoint')
    parser.add_argument('--relation_topk', type=int, default=3, help='Number of top relations used for aggregation/logging')
    parser.add_argument('--relation_weight', type=float, default=1.0, help='Weight for relation score when fusing image scores')
    parser.add_argument('--relation_aggregation', choices=['max', 'topk'], default='topk', help='Aggregation strategy for relation scores')
    parser.add_argument('--relation_min_component_pixels', type=int, default=64, help='Ignore connected components smaller than this value')
    return parser.parse_args()