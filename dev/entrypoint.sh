#!/bin/bash

set -x

# Load the variables defined in the environment file if it is present. Otherwise, skip.
source .env || true

# Run extra commands
exec "$@"