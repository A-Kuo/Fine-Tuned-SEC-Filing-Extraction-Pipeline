"""Poll for an Async Inference result: browser -> this function -> S3.

invoke.py returns an output_location (an s3:// URI) immediately, before
the model has actually run -- SageMaker writes the real result there only
once inference completes, which can be seconds to minutes later depending
on whether the endpoint's GPU instance was already warm or had to scale
up from zero. The frontend polls this endpoint with that output_location
until the object exists.

Required Vercel environment variables: AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_REGION (same as invoke.py).
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps({"error": message}).encode())


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        output_location = (query.get("output_location") or [None])[0]
        if not output_location or not output_location.startswith("s3://"):
            return _error(self, 400, "?output_location=s3://... query param is required")

        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
        bucket, key = output_location[len("s3://"):].split("/", 1)

        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            result = json.loads(obj["Body"].read())
            body = {"status": "complete", "result": result}
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                # Not ready yet -- also check the failure-output path
                # SageMaker Async Inference writes to on a model error, so
                # a genuine failure doesn't poll forever as "pending".
                failure_key = key.replace("async-output/", "async-output/errors/", 1)
                try:
                    err_obj = s3.get_object(Bucket=bucket, Key=failure_key)
                    body = {"status": "failed", "error": err_obj["Body"].read().decode()}
                except ClientError:
                    body = {"status": "pending"}
            else:
                return _error(self, 502, f"S3 error: {e}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
