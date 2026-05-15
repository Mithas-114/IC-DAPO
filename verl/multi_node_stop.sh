#!/bin/bash
# Multi-node Ray cluster stopper
# Usage: bash multi_node_stop.sh MASTER_IP [WORKER_IP1 WORKER_IP2 ...]

set -euo pipefail

RAY_BIN=${RAY_BIN:-$(which ray)}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 MASTER_IP [WORKER_IP1 WORKER_IP2 ...]"
    exit 1
fi

MASTER_IP=$1
shift
WORKER_IPS="$@"

for IP in ${WORKER_IPS}; do
    echo "[INFO] Stopping Ray on worker: ${IP}"
    ssh root@${IP} "${RAY_BIN} stop" || true
done

echo "[INFO] Stopping Ray on master: ${MASTER_IP}"
ssh root@${MASTER_IP} "${RAY_BIN} stop" || true

echo "[INFO] Ray cluster stopped."
