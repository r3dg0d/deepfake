#!/usr/bin/env python3
"""Deliver a session-scoped stop command through Deepfake's existing file IPC."""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'deepfake'
try:
    session = json.loads((root / 'session.json').read_text())
    if not session.get('active') or session.get('session_id') != sys.argv[1]:
        raise ValueError('Session ended or changed; no stop command sent')
    with (root / 'widget-control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'control.json'
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        commands = previous.get('commands', [previous] if previous.get('cmd') else []) if isinstance(previous, dict) else []
        commands.append({'cmd': 'stop', 'session_id': session['session_id']})
        fd, temp = tempfile.mkstemp(prefix='control-', suffix='.tmp', dir=root)
        with os.fdopen(fd, 'w') as file:
            json.dump({'commands': commands}, file)
        os.replace(temp, path)
    print('Stop requested')
except (OSError, ValueError, IndexError) as error:
    print(str(error))
