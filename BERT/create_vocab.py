import argparse
import logging
import sys

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

def main(args):
  logger.info(f"Loading tokenizer from %s", args.model_name_or_path)
  tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

  tokenizer.save_vocabulary(args.vocab_file)
  logger.info("Saved vocabulary file to %s", args.vocab_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Save BERT vocab to disk')
  parser.add_argument(
    '--model_name_or_path',
    help='path to pretrained model or model identifier from which the vocabulary will be created')
  parser.add_argument(
    '--vocab_file',
    help='path to the output vocab file')
  args = parser.parse_args()

  main(args)