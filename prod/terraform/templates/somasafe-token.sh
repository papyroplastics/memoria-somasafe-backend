#!/bin/sh
curl -sf -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
