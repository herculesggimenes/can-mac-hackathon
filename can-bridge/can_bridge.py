"""
python-can compatible interface for the Rust can-bridge daemon.
Communicates over Unix socket with minimal overhead.
"""

import socket
import struct
from typing import Optional

import can

CMD_SEND = 0x01
CMD_RECV = 0x02
FRAME_SIZE = 14


class CanBridgeBus(can.BusABC):
    """python-can Bus implementation backed by the Rust can-bridge daemon."""

    def __init__(self, channel: int = 0, **kwargs):
        super().__init__(channel=channel, **kwargs)
        socket_path = f"/tmp/can{channel}.sock"
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.connect(socket_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{socket_path} not found. Create it with the Rust gs_usb bridge "
                f"`./target/release/can-bridge {channel}` (needs CANable USB IDs 1d50:606f), "
                f"or `can-bridge/slcan_bridge.py --socket {socket_path}` for CDC/SLCAN firmware. "
                f"Bimanual macOS: run `can-bridge/start_bimanual_bridges.sh` "
                f"(second dongle may use SLCAN + YAM_SLCAN_SERIAL if not gs_usb)."
            ) from exc
        self._sock.setblocking(True)
        self.channel_info = f"can-bridge:{channel}"
        self._state = can.BusState.ACTIVE

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def send(self, msg: can.Message, timeout: Optional[float] = None) -> None:
        data = bytearray(FRAME_SIZE)
        data[0] = CMD_SEND
        struct.pack_into("<IB", data, 1, msg.arbitration_id, msg.dlc)
        data[6:6 + len(msg.data)] = msg.data
        self._sock.sendall(data)

    def _recv_internal(self, timeout: Optional[float]):
        try:
            if timeout is not None:
                self._sock.settimeout(timeout)
            else:
                self._sock.settimeout(None)

            buf = b""
            while len(buf) < FRAME_SIZE:
                chunk = self._sock.recv(FRAME_SIZE - len(buf))
                if not chunk:
                    return None, False
                buf += chunk

            if buf[0] != CMD_RECV:
                return None, False

            can_id, dlc = struct.unpack_from("<IB", buf, 1)
            dlc = min(dlc, 8)
            data = buf[6:6 + dlc]
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                dlc=dlc,
                is_extended_id=False,
            )
            return msg, False
        except socket.timeout:
            return None, False

    def shutdown(self) -> None:
        self._sock.close()
