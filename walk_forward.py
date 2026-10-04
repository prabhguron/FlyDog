"""Drive the flydog through the BeagleY-AI server (Freenove's real walking code).

The server needs movement commands repeated every 0.25 s (its watchdog turns the
servos off after 2 s of silence), so this script keeps repeating the command.

  python3 drive.py forward --seconds 5
  python3 drive.py left --seconds 3
  python3 drive.py forward --seconds 5 --speed 6 --host flydog.local   (from your Mac)

Ctrl+C sends stop.
"""
import argparse
import socket
import time

COMMANDS = {
    'forward': 'CMD_MOVE_FORWARD',
    'backward': 'CMD_MOVE_BACKWARD',
    'left': 'CMD_TURN_LEFT',          # turn on the spot
    'right': 'CMD_TURN_RIGHT',
    'stepleft': 'CMD_MOVE_LEFT',      # side-step
    'stepright': 'CMD_MOVE_RIGHT',
}
STOP = 'CMD_MOVE_STOP'
REPEAT = 0.2      # seconds between repeats (server wants at least every 0.25 s)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('move', choices=sorted(COMMANDS))
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--speed', type=int, default=8, help='Freenove speed value (default 8)')
    parser.add_argument('--host', default='127.0.0.1', help='127.0.0.1 on the dog, flydog.local from the Mac')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    command = f'{COMMANDS[args.move]}#{args.speed}\n'.encode()
    sock = None
    for attempt in range(20):              # server may still be starting up
        try:
            sock = socket.create_connection((args.host, args.port), timeout=2)
            break
        except OSError:
            time.sleep(0.5)
    if sock is None:
        raise SystemExit(f'Could not connect to the server on {args.host}:{args.port} after 10 s')
    print(f'Connected. Sending {COMMANDS[args.move]}#{args.speed} for {args.seconds}s. Ctrl+C to stop.')
    try:
        end = time.time() + args.seconds
        while time.time() < end:
            sock.sendall(command)
            time.sleep(REPEAT)
    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        try:
            sock.sendall(f'{STOP}\n'.encode())
            time.sleep(0.2)
        finally:
            sock.close()
    print('Sent stop. (On this server, stop turns the servos off - the dog will relax.)')


if __name__ == '__main__':
    main()