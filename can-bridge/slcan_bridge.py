#!/usr/bin/env python3
"""SLCAN fallback bridge for CANable serial firmware on macOS.

It exposes the same Unix socket protocol as the Rust gs_usb bridge:
  send: [0x01][can_id u32 LE][dlc u8][8 data bytes]
  recv: [0x02][can_id u32 LE][dlc u8][8 data bytes]
"""

from __future__ import annotations

import argparse
import os
import selectors
import socket
import struct
import sys
import time

import serial

CMD_SEND = 0x01
CMD_RECV = 0x02
FRAME_SIZE = 14

BITRATE_COMMANDS = {
    10_000: "S0",
    20_000: "S1",
    50_000: "S2",
    100_000: "S3",
    125_000: "S4",
    250_000: "S5",
    500_000: "S6",
    750_000: "S7",
    1_000_000: "S8",
}


def write_cmd(ser: serial.Serial, cmd: str) -> None:
    ser.write(cmd.encode("ascii") + b"\r")
    ser.flush()


def configure_slcan(ser: serial.Serial, bitrate: int) -> None:
    if bitrate not in BITRATE_COMMANDS:
        raise ValueError(f"Unsupported SLCAN bitrate: {bitrate}")
    write_cmd(ser, "C")
    time.sleep(0.05)
    ser.reset_input_buffer()
    write_cmd(ser, BITRATE_COMMANDS[bitrate])
    write_cmd(ser, "M0")  # normal mode
    write_cmd(ser, "A1")  # auto retransmit
    write_cmd(ser, "O")
    time.sleep(0.05)


def encode_slcan_frame(can_id: int, dlc: int, data: bytes) -> bytes:
    data_hex = data[:dlc].hex().upper()
    if can_id <= 0x7FF:
        return f"t{can_id:03X}{dlc:X}{data_hex}\r".encode("ascii")
    return f"T{can_id:08X}{dlc:X}{data_hex}\r".encode("ascii")


def decode_slcan_line(line: bytes) -> tuple[int, int, bytes] | None:
    if not line:
        return None
    text = line.decode("ascii", errors="ignore").strip()
    if not text or text[0] not in {"t", "T"}:
        return None
    try:
        if text[0] == "t":
            can_id = int(text[1:4], 16)
            dlc = int(text[4], 16)
            offset = 5
        else:
            can_id = int(text[1:9], 16)
            dlc = int(text[9], 16)
            offset = 10
        data = bytes.fromhex(text[offset : offset + dlc * 2])
    except (ValueError, IndexError):
        return None
    return can_id, dlc, data


def socket_payload(can_id: int, dlc: int, data: bytes) -> bytes:
    payload = bytearray(FRAME_SIZE)
    payload[0] = CMD_RECV
    struct.pack_into("<IB", payload, 1, can_id, dlc)
    payload[6 : 6 + min(dlc, 8)] = data[:8]
    return bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/cu.usbmodem206D338A594E1")
    parser.add_argument("--socket", default="/tmp/can0.sock")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--tty-baudrate", type=int, default=115_200)
    args = parser.parse_args()

    try:
        os.remove(args.socket)
    except FileNotFoundError:
        pass

    ser = serial.Serial(args.serial, args.tty_baudrate, timeout=0)
    configure_slcan(ser, args.bitrate)
    print(f"SLCAN bridge open: {args.serial} @ CAN {args.bitrate}, socket {args.socket}", flush=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(1)
    try:
        while True:
            conn, _ = server.accept()
            conn.setblocking(False)
            print("Client connected", flush=True)

            selector = selectors.DefaultSelector()
            selector.register(conn, selectors.EVENT_READ)
            selector.register(ser.fileno(), selectors.EVENT_READ)
            serial_buf = b""

            try:
                while True:
                    for key, _ in selector.select(timeout=0.05):
                        if key.fileobj is conn:
                            packet = conn.recv(FRAME_SIZE)
                            if not packet:
                                raise ConnectionResetError
                            if len(packet) != FRAME_SIZE or packet[0] != CMD_SEND:
                                continue
                            can_id, dlc = struct.unpack_from("<IB", packet, 1)
                            data = packet[6 : 6 + min(dlc, 8)]
                            ser.write(encode_slcan_frame(can_id, dlc, data))
                        else:
                            serial_buf += ser.read(4096)
                            while b"\r" in serial_buf:
                                line, serial_buf = serial_buf.split(b"\r", 1)
                                decoded = decode_slcan_line(line)
                                if decoded is None:
                                    continue
                                conn.sendall(socket_payload(*decoded))
            except (BrokenPipeError, ConnectionResetError):
                conn.close()
                selector.close()
                print("Client disconnected", flush=True)
                continue
    finally:
        write_cmd(ser, "C")
        ser.close()
        server.close()
        try:
            os.remove(args.socket)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
