import sys
from app.config import Configs
from app.logger import Logger
from app.train import TurkishClassification
from app.inference import Inference
from app.logger import Logger


def main(args, configs):

    logger = Logger(**configs["logger"])

    logger.debug("############ TURKISH SENTIMENT CONFIGURATIONS ############")
    logger.debug(configs)


    if args.test:
        sys.exit()
    
    if args.train:
        sentiment_trainer = TurkishClassification(**configs["Train"],logger = logger)
        sentiment_trainer.train()

    if args.infer:
        sentiment_infer = Inference(**configs["Inference"],logger = logger)
        sentiment_infer.infer()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    parser.add_argument("--test", action= "store_true")
    parser.add_argument("--infer", action= "store_true")
    parser.add_argument("--train", action= "store_true")


    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
