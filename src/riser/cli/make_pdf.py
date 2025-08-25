#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules


#################### ARGUMENT PARSER ####################
Description = """."""

Examples = """Examples:
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")

    output_args = parser.add_argument_group("Outputs")

    return parser.parse_args(args=iargs)



#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()


if __name__ == '__main__':
    main()


# end of file