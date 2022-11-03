import json

from server.payload.SR.Request import TaskRequest

from .mock import prepare_requests, prepare_setting


def test_serialization():
    requests, _ = prepare_requests()

    for request in requests:
        request_json_str = request.json()

        request_json2 = json.loads(request_json_str)

        request2 = TaskRequest(**request_json2)

        assert request == request2
