#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.


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