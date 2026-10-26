"""Compatibility entrypoint for the current no-filter full-data OOF experiment."""
from pricing.train import run,parser
if __name__=='__main__':run(parser().parse_args())
