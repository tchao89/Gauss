
import argparse
import AutoModelingTree


def main(config="./config.yaml"):
     conf = Config.read(config)
     inference = Inference("inference",
                           conf.work_root,
                           conf.out_put_path):
     inference.run()
if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")
   
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()