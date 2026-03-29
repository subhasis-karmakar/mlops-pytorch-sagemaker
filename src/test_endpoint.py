import json
import boto3

REGION = "us-west-2"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"
PAYLOAD_FILE = "payload.json"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def main():
    with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
        payload = f.read()

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )

    body = response["Body"].read().decode("utf-8")
    print(body)

    parsed = json.loads(body)
    if "predictions" not in parsed:
        raise RuntimeError("Endpoint response does not contain 'predictions'")

    print("Endpoint smoke test passed.")


if __name__ == "__main__":
    main()