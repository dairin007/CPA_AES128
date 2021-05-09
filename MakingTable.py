import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Tuple

import numpy as np
import psutil
from tqdm import tqdm

import aes_state
from aes_state import State

__all__ = ["State"]


InvSbox: Tuple[int, ...] = (
    0x52,
    0x09,
    0x6A,
    0xD5,
    0x30,
    0x36,
    0xA5,
    0x38,
    0xBF,
    0x40,
    0xA3,
    0x9E,
    0x81,
    0xF3,
    0xD7,
    0xFB,
    0x7C,
    0xE3,
    0x39,
    0x82,
    0x9B,
    0x2F,
    0xFF,
    0x87,
    0x34,
    0x8E,
    0x43,
    0x44,
    0xC4,
    0xDE,
    0xE9,
    0xCB,
    0x54,
    0x7B,
    0x94,
    0x32,
    0xA6,
    0xC2,
    0x23,
    0x3D,
    0xEE,
    0x4C,
    0x95,
    0x0B,
    0x42,
    0xFA,
    0xC3,
    0x4E,
    0x08,
    0x2E,
    0xA1,
    0x66,
    0x28,
    0xD9,
    0x24,
    0xB2,
    0x76,
    0x5B,
    0xA2,
    0x49,
    0x6D,
    0x8B,
    0xD1,
    0x25,
    0x72,
    0xF8,
    0xF6,
    0x64,
    0x86,
    0x68,
    0x98,
    0x16,
    0xD4,
    0xA4,
    0x5C,
    0xCC,
    0x5D,
    0x65,
    0xB6,
    0x92,
    0x6C,
    0x70,
    0x48,
    0x50,
    0xFD,
    0xED,
    0xB9,
    0xDA,
    0x5E,
    0x15,
    0x46,
    0x57,
    0xA7,
    0x8D,
    0x9D,
    0x84,
    0x90,
    0xD8,
    0xAB,
    0x00,
    0x8C,
    0xBC,
    0xD3,
    0x0A,
    0xF7,
    0xE4,
    0x58,
    0x05,
    0xB8,
    0xB3,
    0x45,
    0x06,
    0xD0,
    0x2C,
    0x1E,
    0x8F,
    0xCA,
    0x3F,
    0x0F,
    0x02,
    0xC1,
    0xAF,
    0xBD,
    0x03,
    0x01,
    0x13,
    0x8A,
    0x6B,
    0x3A,
    0x91,
    0x11,
    0x41,
    0x4F,
    0x67,
    0xDC,
    0xEA,
    0x97,
    0xF2,
    0xCF,
    0xCE,
    0xF0,
    0xB4,
    0xE6,
    0x73,
    0x96,
    0xAC,
    0x74,
    0x22,
    0xE7,
    0xAD,
    0x35,
    0x85,
    0xE2,
    0xF9,
    0x37,
    0xE8,
    0x1C,
    0x75,
    0xDF,
    0x6E,
    0x47,
    0xF1,
    0x1A,
    0x71,
    0x1D,
    0x29,
    0xC5,
    0x89,
    0x6F,
    0xB7,
    0x62,
    0x0E,
    0xAA,
    0x18,
    0xBE,
    0x1B,
    0xFC,
    0x56,
    0x3E,
    0x4B,
    0xC6,
    0xD2,
    0x79,
    0x20,
    0x9A,
    0xDB,
    0xC0,
    0xFE,
    0x78,
    0xCD,
    0x5A,
    0xF4,
    0x1F,
    0xDD,
    0xA8,
    0x33,
    0x88,
    0x07,
    0xC7,
    0x31,
    0xB1,
    0x12,
    0x10,
    0x59,
    0x27,
    0x80,
    0xEC,
    0x5F,
    0x60,
    0x51,
    0x7F,
    0xA9,
    0x19,
    0xB5,
    0x4A,
    0x0D,
    0x2D,
    0xE5,
    0x7A,
    0x9F,
    0x93,
    0xC9,
    0x9C,
    0xEF,
    0xA0,
    0xE0,
    0x3B,
    0x4D,
    0xAE,
    0x2A,
    0xF5,
    0xB0,
    0xC8,
    0xEB,
    0xBB,
    0x3C,
    0x83,
    0x53,
    0x99,
    0x61,
    0x17,
    0x2B,
    0x04,
    0x7E,
    0xBA,
    0x77,
    0xD6,
    0x26,
    0xE1,
    0x69,
    0x14,
    0x63,
    0x55,
    0x21,
    0x0C,
    0x7D,
)
WaveNum = 10000


def MakingTable() -> None:
    text: List[str] = np.loadtxt(
        "./InputData/CIPHERTEXT10000.txt", dtype="U", max_rows=WaveNum
    )
    TextNum: int = len(text)
    args: List[Any] = [0] * 16
    ProcessNum: int = 0

    for column in range(4):
        for row in range(4):
            arg: List[Any] = [0]

            arg = [row, column, TextNum, text]
            args[row + 4 * column] = arg
    ProcessNum = psutil.cpu_count()
    with ProcessPoolExecutor(max_workers=ProcessNum) as executor:
        list(tqdm(executor.map(ByteTable, args), total=16))


def ByteTable(args: List[Any]) -> None:
    row: int = args[0]
    column: int = args[1]
    TextNum: int = args[2]
    text: List[str] = args[3]

    HDTable: List[List[int]] = [[0]] * TextNum
    for i, CipherText in enumerate(text):
        CipherSt: aes_state.State = aes_state.State([])
        KeySt: aes_state.State = aes_state.State([])

        CipherSt = InputState(CipherText)
        HD: List[int] = [0] * 256
        for Key in range(256):
            R9OutSt: aes_state.State = aes_state.State([])
            ShiftCipherSt: aes_state.State = aes_state.State([])
            hd: int = 0

            KeySt.set(row, column, Key)
            R9OutSt = R10toR9(CipherSt, KeySt)
            ShiftCipherSt = ShiftRows(CipherSt)
            hd = HDCalc(row, column, ShiftCipherSt, R9OutSt)
            HD[Key] = hd
        HDTable[i] = HD
    np.savetxt(
        "./Table/S_{}_{}.csv".format(row, column),
        HDTable,
        delimiter=",",
        fmt="%d",
    )


def R10toR9(R10St: aes_state.State, KeySt: aes_state.State) -> aes_state.State:
    InvAddRoundSt: aes_state.State = aes_state.State([])
    InvSubSt: aes_state.State = aes_state.State([])

    InvAddRoundSt = AddRoundKey(R10St, KeySt)
    InvSubSt = InvSubByte(InvAddRoundSt)
    return InvSubSt


def InputState(text: str) -> aes_state.State:
    assert len(text) == 34, ValueError("Input Error")
    ResSt: aes_state.State = aes_state.State([])
    ByteList: List[str] = []

    ByteList = re.split("(..)", text)[1::2]
    ResSt = aes_state.State(list(map(lambda x: int(x, 16), ByteList[1:])))
    return ResSt


def ShiftRows(InputSt: aes_state.State) -> aes_state.State:
    ResSt: aes_state.State = aes_state.State([])

    ResSt.set(0, 0, InputSt.get(0, 0))
    ResSt.set(0, 1, InputSt.get(0, 1))
    ResSt.set(0, 2, InputSt.get(0, 2))
    ResSt.set(0, 3, InputSt.get(0, 3))

    ResSt.set(1, 0, InputSt.get(1, 1))
    ResSt.set(1, 1, InputSt.get(1, 2))
    ResSt.set(1, 2, InputSt.get(1, 3))
    ResSt.set(1, 3, InputSt.get(1, 0))

    ResSt.set(2, 0, InputSt.get(2, 2))
    ResSt.set(2, 1, InputSt.get(2, 3))
    ResSt.set(2, 2, InputSt.get(2, 0))
    ResSt.set(2, 3, InputSt.get(2, 1))

    ResSt.set(3, 0, InputSt.get(3, 3))
    ResSt.set(3, 1, InputSt.get(3, 0))
    ResSt.set(3, 2, InputSt.get(3, 1))
    ResSt.set(3, 3, InputSt.get(3, 2))
    return ResSt


def InvShiftRows(InputSt: aes_state.State) -> aes_state.State:
    ResSt: aes_state.State = aes_state.State([])

    ResSt.set(0, 0, InputSt.get(0, 0))
    ResSt.set(0, 1, InputSt.get(0, 1))
    ResSt.set(0, 2, InputSt.get(0, 2))
    ResSt.set(0, 3, InputSt.get(0, 3))

    ResSt.set(1, 0, InputSt.get(1, 3))
    ResSt.set(1, 1, InputSt.get(1, 0))
    ResSt.set(1, 2, InputSt.get(1, 1))
    ResSt.set(1, 3, InputSt.get(1, 2))

    ResSt.set(2, 0, InputSt.get(2, 2))
    ResSt.set(2, 1, InputSt.get(2, 3))
    ResSt.set(2, 2, InputSt.get(2, 0))
    ResSt.set(2, 3, InputSt.get(2, 1))

    ResSt.set(3, 0, InputSt.get(3, 1))
    ResSt.set(3, 1, InputSt.get(3, 2))
    ResSt.set(3, 2, InputSt.get(3, 3))
    ResSt.set(3, 3, InputSt.get(3, 0))
    return ResSt


def InvS_BOX(Inb: int) -> int:
    return InvSbox[Inb]


def InvSubByte(InputSt: aes_state.State) -> aes_state.State:
    ResSt: aes_state.State = aes_state.State([])

    for row in range(4):
        for column in range(4):
            value: int = 0

            value = InvS_BOX(InputSt.get(row, column))
            ResSt.set(row, column, value)
    return ResSt


def AddRoundKey(InputSt: aes_state.State, key: aes_state.State) -> aes_state.State:
    ResSt: aes_state.State = aes_state.State([])

    for row in range(4):
        for column in range(4):
            value: str = ""

            value = "{:016X}".format(InputSt.get(row, column) ^ key.get(row, column))
            ResSt.set(row, column, int(value[-2:], 16))
    return ResSt


def HDCalc(row: int, column: int, St1: aes_state.State, St2: aes_state.State) -> int:
    hd: int = 0

    hd = (bin(St1.get(row, column) ^ St2.get(row, column))).count("1")
    return hd


MakingTable()
