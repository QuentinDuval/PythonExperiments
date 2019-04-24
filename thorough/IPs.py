"""
https://leetcode.com/problems/validate-ip-address
"""


class Solution:
    def validIPAddress(self, ip: str) -> str:
        if self.validIPv4(ip):
            return "IPv4"
        elif self.validIPv6(ip):
            return "IPv6"
        return "Neither"

    def validIPv4(self, ip: str) -> bool:
        parts = ip.split('.')
        return len(parts) == 4 and all(self.validTokenV4(part) for part in parts)

    def validTokenV4(self, token: str) -> bool:
        if not token: return False
        if len(token) > 3: return False
        if token == '0': return True
        if token[0] == '0': return False
        if not all(c.isdigit() for c in token): return False
        return int(token) <= 255

    def validIPv6(self, ip: str) -> str:
        parts = ip.split(':')
        return len(parts) == 8 and all(self.validTokenV6(part) for part in parts)

    def validTokenV6(self, token: str) -> bool:
        if not token: return False
        if len(token) > 4: return False
        if not all(c.isdigit() or c in 'ABCDEFabcdef' for c in token): return False
        return True
