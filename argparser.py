import argparse
from typing import Iterable, List


def _parse_hidden_dims(value: Iterable[int] | str) -> List[int]:
    """Parses comma-separated integers provided via the CLI."""

    if isinstance(value, (list, tuple)):
        dims = [int(v) for v in value]
    else:
        parts = [part.strip() for part in str(value).split(',')]
        dims = [int(part) for part in parts if part]
    if not dims:
        raise argparse.ArgumentTypeError("--relation-hidden-dims requires at least one integer")
    return dims


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

    # Relation branch specific arguments
    parser.add_argument('--relation-class-embedding-dim', type=int, default=32,
                        help='Embedding size used for class ids in the relation branch MLP.')
    parser.add_argument('--relation-hidden-dims', default='128,64',
                        help='Comma separated hidden dimensions for the relation branch MLP.')
    parser.add_argument('--relation-dropout', type=float, default=0.1,
                        help='Dropout applied between relation branch MLP layers.')
    parser.add_argument('--relation-min-component-area', type=int, default=10,
                        help='Minimum number of pixels per component used by the relation branch.')
    parser.add_argument('--relation-component-connectivity', type=int, choices=[1, 2], default=1,
                        help='Connectivity used when extracting connected components for the relation branch.')
    parser.add_argument('--relation-score-reduction', choices=['mean', 'max', 'sum'], default='mean',
                        help='Reduction strategy applied to per-pair scores at inference time.')
    parser.add_argument('--relation-loss-weight', type=float, default=1.0,
                        help='Scalar applied to the relation branch loss before summing it into the total loss.')

    args = parser.parse_args()
    args.relation_hidden_dims = tuple(_parse_hidden_dims(args.relation_hidden_dims))
    return args
