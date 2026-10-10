#!/bin/sh
set -e
dir=$(cd "$(dirname "$0")" && pwd)
backend=$(cd "$dir/.." && pwd)
registry=$(terraform -chdir="$dir/terraform" output -raw registry)
project=$(terraform -chdir="$dir/terraform" output -raw project)

make -C "$backend" prod-build
gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin "${registry%%/*}"
for image in api worker bench; do
  podman tag "localhost/somasafe-$image:latest" "$registry/somasafe-$image:latest"
  podman push "$registry/somasafe-$image:latest"
done
gcloud secrets versions add server-private-key --project "$project" \
  --data-file="$backend/shared/gen/server-private-key.pem"

until make -C "$backend" prod-db-seed ARGS="${1:-200}"; do
  echo "seeding failed, retrying in 15s (is connect.sh running?)"
  sleep 15
done
