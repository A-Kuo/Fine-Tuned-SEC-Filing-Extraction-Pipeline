"""Thin proxy: browser -> this function -> SageMaker Async Inference.

Why this exists at all: SageMaker's InvokeEndpoint API requires AWS
Signature Version 4 request signing. A browser cannot do that with a plain
fetch() -- there is no way to safely hold AWS credentials client-side to
sign with. This function holds the credentials server-side (as Vercel
environment variables, never sent to the browser) and does the signed
call on the frontend's behalf.

Uses Vercel's plain BaseHTTPRequestHandler convention (auto-detected by
file path under api/, no framework/pyproject.toml entrypoint config
needed) -- see https://vercel.com/docs/functions/runtimes/python/api-directory.

Async Inference contract: invoke_endpoint_async needs the request payload
already sitting in S3 (InputLocation), not inline in the API call, and
returns immediately with an OutputLocation the result will eventually be
written to -- it does not block until the model responds. The frontend
must poll status.py's endpoint with that OutputLocation until the result
appears (a cold-started GPU instance can take minutes to spin up, well
beyond what a single Vercel function invocation should block on).

Required Vercel environment variables (set via the dashboard, never
committed): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
SAGEMAKER_ENDPOINT_NAME, SAGEMAKER_S3_BUCKET.
"""

import json
import os
import uuid
from http.server import BaseHTTPRequestHandler


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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
        bucket = os.environ.get("SAGEMAKER_S3_BUCKET")
        if not endpoint_name or not bucket:
            return _error(self, 503, "SAGEMAKER_ENDPOINT_NAME/SAGEMAKER_S3_BUCKET not configured on this deployment")

        content_length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            return _error(self, 400, "Request body must be valid JSON")

        if "task" not in payload:
            return _error(self, 400, "Request body must include a 'task' field ('extract' or 'rag_query')")

        # Imported here, not at module top level, so a misconfigured/
        # dependency-missing deployment fails per-request with a clear
        # error rather than making the whole function fail to even load.
        import boto3

        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
        sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION"))

        request_id = str(uuid.uuid4())
        input_key = f"findoc-llama-sec/async-input/{request_id}.json"
        s3.put_object(Bucket=bucket, Key=input_key, Body=json.dumps(payload).encode(), ContentType="application/json")
        input_location = f"s3://{bucket}/{input_key}"

        try:
            response = sagemaker_runtime.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InputLocation=input_location,
                ContentType="application/json",
            )
        except Exception as e:
            return _error(self, 502, f"SageMaker invoke failed: {e}")

        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "pending",
            "inference_id": response.get("InferenceId"),
            "output_location": response.get("OutputLocation"),
        }).encode())
