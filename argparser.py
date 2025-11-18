import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='screw_bag', help='category')
    parser.add_argument('-o', '--output_dir', default='./results/')
    parser.add_argument('-w', '--weights', default='models/teacher_medium.pth')
    parser.add_argument('-i', '--imagenet_train_path', default='./data/imagenet/train')
    parser.add_argument('--mvtec_loco_path', default='./data/mvtec_loco')
    parser.add_argument('--mvtec_loco_seg_path', default='./data/mvtec_loco_composition_maps/')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_relation_branch', action='store_true',
                        help='Enable the relation learning branch during training.')
    parser.add_argument('--relation_hidden_dim', type=int, default=128,
                        help='Hidden dimension for the relation branch MLP.')
    parser.add_argument('--relation_embedding_dim', type=int, default=64,
                        help='Embedding dimension for relation class embeddings.')
    parser.add_argument('--relation_lambda', type=float, default=0.1,
                        help='Scaling factor for the relation branch loss.')
    parser.add_argument('--relation_max_pairs', type=int, default=128,
                        help='Maximum number of positive/negative pairs sampled per batch.')
    parser.add_argument('--relation_num_classes', type=int, default=6,
                        help='Number of composition classes for the relation branch embeddings.')
    parser.add_argument('--resume_from', default='',
                        help='Optional path to a directory containing *_tmp.pth checkpoints to resume from.')
    return parser.parse_args()
