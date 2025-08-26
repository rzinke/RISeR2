#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Driver
def test():
    """Sanity check: attempt to access the package.
    """
    # Import module
    import riser

    return 0


# Bootstrap
if __name__ == "__main__":
    # Invoke the driver
    status = test()

    # Share the status with the shell
    raise SystemExit(status)


# end of file