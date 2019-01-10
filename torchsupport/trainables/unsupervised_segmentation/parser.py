from argparse import ArgumentParser

def parse():
  parser = ArgumentParser(description="unsupervised semantic segmentation.")

  # general training settings:
  parser.add_argument('--path', type=str, required=True, help="path to input images.")
  parser.add_argument('--data_type', type=str, default="ndpi", help="input image type.")
  parser.add_argument('--train', action="store_true", help="train on input images?")
  parser.add_argument('--eval', action="store_true", help="evaluate network on input images?")
  parser.add_argument('--cuda', action="store_true", help="run using cuda?")
  parser.add_argument('--on_gpu', type=int, default=0, help="bind to given GPU.")
  parser.add_argument('--batch', type=int, default=64, help="training batch size.")
  parser.add_argument('--epochs', type=int, default=50, help="number of training epochs.")
  parser.add_argument('--lr', type=float, default=0.001, help="learning rate.")
  parser.add_argument('--threads', type=int, default=16, help="data loader threads.")
  parser.add_argument('--seed', type=int, default=42, help="random seed, defaults to 42.")
  
  # general arch settings:
  parser.add_argument('--unsupervision', type=str, default="SegmenterDecoder",
                      choices=["SegmenterDecoder", "ResidualDecoder", "MultiDecoder"],
                      help="type of unsupervised architecture. Defaults to a WNet-style segmenter-decoder.")
  parser.add_argument('--arch', type=str, default="UNet", choices=["UNet", "AutofocusNet", "SplitNet"],
                      help="architecture to be used. Defaults to UNet.")
  parser.add_argument('--regularization', type=str, default="all",
                      help="regularization for more natural segmentation.")
  parser.add_argument('--max_classes', type=int, default=4, help="maximum number of different classes.")

  # UNet settings:
  parser.add_argument('--depth', type=int, default=4, help="UNet depth.")
  parser.add_argument('--multiscale', action="store_true", help="use multi-scale model?")
  parser.add_argument('--dilate', action="store_true", help="use dilated convolutions?")
  parser.add_argument('--attention', action="store_true", help="use attention gate?")

  opt = parser.parse_args()
  return opt
