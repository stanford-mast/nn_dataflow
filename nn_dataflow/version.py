""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

import hashlib
import os
import string
import subprocess

from . import __version__

def _command_output(args, cwd):
    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE).stdout as fh:
        return fh.read().strip()

def get_version(with_local=False):
    ''' Get the version number, optionally with the local version number. '''

    version = __version__

    if with_local:
        cwd = os.path.dirname(os.path.abspath(__file__))

        if subprocess.call(['git', 'rev-parse'], cwd=cwd,
                           stderr=subprocess.STDOUT,
                           stdout=open(os.devnull, 'w')) != 0:
            # Not in git repo.
            return version  # pragma: no cover

        # Dirty summary.
        short_stat = _command_output(['git', 'diff', 'HEAD', '--shortstat'],
                                     cwd) \
                .replace('files changed', 'fc').replace('file changed', 'fc') \
                .replace('insertions(+)', 'a').replace(' insertion(+)', 'a') \
                .replace('deletions(-)', 'd').replace(' deletion(-)', 'd') \
                .replace(',', '').replace(' ', '')
        diff_hash = hashlib.md5(_command_output(['git', 'diff', 'HEAD'], cwd)) \
                .hexdigest()[:8]
        dirty = '' if not short_stat else '-' + short_stat + '-' + diff_hash

        # Git describe.
        desc = _command_output(['git', 'describe', '--tags', '--always',
                                '--dirty={}'.format(dirty)],
                               cwd)

        version += '+' + desc

    assert not any(w in version for w in string.whitespace)
    return version

