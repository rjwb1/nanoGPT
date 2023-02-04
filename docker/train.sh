#!/usr/bin/env bash
set -e
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"

# Run backend
docker-compose run --rm nanogpt
