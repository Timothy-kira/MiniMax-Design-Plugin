"""Check that local progress reads can pass a pending cloud operation."""

import io
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import server


class LocalReadTests(unittest.TestCase):
    def test_status_responds_while_cloud_operation_is_blocked(self):
        started = threading.Event()
        release = threading.Event()
        requests = [
            {"id": 1, "method": "tools/call", "params": {"name": "review_runtime"}},
            {"id": 2, "method": "tools/call", "params": {"name": "video_status"}},
        ]

        def dispatch(packet):
            if packet["id"] == 1:
                started.set()
                if not release.wait(2):
                    raise ValueError("Local status was blocked behind the cloud request")
            else:
                self.assertTrue(started.wait(2))
            return {"test_id": packet["id"]}

        class Output(io.BytesIO):
            def write(self, data):
                count = super().write(data)
                if json.loads(data)["id"] == 2:
                    release.set()
                return count

        inp = io.BytesIO(b"".join(json.dumps(x).encode() + b"\n" for x in requests))
        out = Output()
        with (patch.object(server, "dispatch", side_effect=dispatch),
              patch.object(server.controller, "check_cli_update"),
              patch.object(server.sys, "stdin", SimpleNamespace(buffer=inp)),
              patch.object(server.sys, "stdout", SimpleNamespace(buffer=out))):
            server.main()
        self.assertEqual([json.loads(line)["id"] for line in out.getvalue().splitlines()], [2, 1])
