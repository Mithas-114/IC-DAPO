#!/bin/bash
# Multi-node Ray cluster launcher
# Usage: bash multi_node_launcher.sh MASTER_IP [WORKER_IP1 WORKER_IP2 ...]
# Example: bash multi_node_launcher.sh 192.168.1.100 192.168.1.101 192.168.1.102
#
# Environment variables (optional):
#   RAY_PORT        - Ray head port (default: 6379)
#   DASHBOARD_PORT  - Dashboard port (default: 8265)
#   RAY_BIN         - Path to ray binary (default: auto-detect via `which ray`)

set -euo pipefail

RAY_PORT=${RAY_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
RAY_BIN=${RAY_BIN:-$(which ray)}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 MASTER_IP [WORKER_IP1 WORKER_IP2 ...]"
    exit 1
fi

MASTER_IP=$1
shift
WORKER_IPS="$@"

echo "[INFO] Starting Ray head on ${MASTER_IP} ..."
ssh root@${MASTER_IP} "${RAY_BIN} start --head \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --port=${RAY_PORT}"

for IP in ${WORKER_IPS}; do
    echo "[INFO] Starting Ray worker on ${IP} ..."
    ssh root@${IP} "until ${RAY_BIN} start \
        --address=\"${MASTER_IP}:${RAY_PORT}\" \
        --disable-usage-stats; do sleep 2; done"
done

echo "[INFO] Ray cluster started. Head: ${MASTER_IP}:${RAY_PORT}"
