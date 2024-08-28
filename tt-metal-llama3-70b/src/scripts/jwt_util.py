#!/usr/bin/env python3
import argparse
import json
import os

import sys

import jwt


def main():
    parser = argparse.ArgumentParser(description="Generate signed JWT payload")
    parser.add_argument("mode", type=str, help="'encode' or 'decode'")
    parser.add_argument(
        "payload", type=str, help="JSON string if 'encode', token if 'decode'"
    )
    parser.add_argument(
        "--secret",
        type=str,
        dest="secret",
        help="JWT secret if not provided as environment variable JWT_SECRET",
    )
    args = parser.parse_args()

    try:
        jwt_secret = os.environ.get("JWT_SECRET", args.secret)
    except KeyError:
        print("ERROR: Expected JWT_SECRET environment variable to be provided")
        sys.exit(1)

    try:
        if args.mode == "encode":
            json_payload = json.loads(args.payload)
            encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
            print(encoded_jwt)
        elif args.mode == "decode":
            decoded_jwt = jwt.decode(args.payload, jwt_secret, algorithms="HS256")
            print(decoded_jwt)
        else:
            print("ERROR: Expected mode to be 'encode' or 'decode'")
            sys.exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Expected payload to be a valid JSON string")
        sys.exit(1)


if __name__ == "__main__":
    main()