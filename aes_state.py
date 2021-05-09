from typing import List


class State:
    # AES state
    # r:row,c:column

    def __init__(self, b: List[int]) -> None:
        self.s: List[int] = [0] * 16
        for i in range(min(16, len(b))):
            self.s[i] = b[i]

    def get(self, r: int, c: int) -> int:
        return self.s[r + 4 * c]

    def get_byte(self) -> List[int]:
        return self.s

    def set(self, r: int, c: int, value: int) -> None:
        self.s[r + 4 * c] = value

    def s_print(self) -> str:
        res: str = ""
        for r in range(4):
            res += "-------------\r\n"
            res += "|"
            for c in range(4):
                res += "{:02x}".format(self.get(r, c)) + "|"
            res += "\r\n"
        res += "-------------\r\n"
        return res
