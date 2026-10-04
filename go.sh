#!/bin/bash
# go.sh - stand the dog up, walk, then stand him back up.
#
#   bash ~/flydog/go.sh                 # walk forward 2 seconds
#   bash ~/flydog/go.sh 5               # walk forward 5 seconds
#   bash ~/flydog/go.sh 3 left          # turn left 3 seconds
#   moves: forward backward left right stepleft stepright
#
# Ctrl+C at any time stops walking and stands him back up.

SECONDS_TO_WALK=${1:-2}
MOVE=${2:-forward}
PY=/home/robot/flydog/.venv/bin/python
CAL=/home/robot/flydog/foot_calibration.py
DRIVE=/home/robot/flydog/walk_forward.py
SERVER=/home/robot/flydog-kit/Code/BeagleYAI/beagle_server.py
SERVER_DIR=/home/robot/flydog-kit/Code/Server
LOG=/home/robot/flydog/server.log

free_port() {
    # stop ANY process holding port 5001 (old servers that ignore a normal stop too)
    sudo pkill -f beagle_server.py 2>/dev/null
    for i in $(seq 1 10); do
        sudo ss -ltn | grep -q ':5001 ' || return 0
        sleep 0.3
    done
    for pid in $(sudo ss -ltnp | grep ':5001 ' | grep -o 'pid=[0-9]*' | cut -d= -f2); do
        echo "Force-stopping leftover process $pid on port 5001"
        sudo kill -9 "$pid"
    done
    sleep 0.5
}

stand_up() {
    echo "== Standing up =="
    free_port
    sudo systemctl start flydog-servo-center
    sleep 1
    sudo "$PY" "$CAL" --neutral
}

finish() {
    trap - INT TERM
    echo
    stand_up
    echo "== Done. He's standing. =="
    exit 0
}
trap finish INT TERM

sudo -v || exit 1          # ask for the password once, up front

# 1. Stand up from whatever state he's in
stand_up

# 2. Start the walking server in the background
echo "== Starting server (log: $LOG) =="
sudo systemctl stop flydog-servo-center
cd "$SERVER_DIR" || exit 1
sudo "$PY" "$SERVER" --enable-motion --no-imu --no-camera > "$LOG" 2>&1 &

sleep 1
if grep -q "Address already in use" "$LOG"; then
    echo "Port 5001 still busy:"; sudo ss -ltnp | grep ':5001 '
    finish
fi
for i in $(seq 1 20); do               # wait up to 10 s for it to listen
    if ss -ltn | grep -q ':5001 '; then break; fi
    sleep 0.5
done
if ! ss -ltn | grep -q ':5001 '; then
    echo "Server didn't start. Last lines of $LOG:"
    tail -20 "$LOG"
    finish
fi

# 3. Walk
echo "== $MOVE for $SECONDS_TO_WALK s =="
python3 "$DRIVE" "$MOVE" --seconds "$SECONDS_TO_WALK"

# 4. Stand back up instead of collapsing
finish