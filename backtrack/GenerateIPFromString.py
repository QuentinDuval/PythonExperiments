"""
https://practice.geeksforgeeks.org/problems/generate-ip-addresses/1

Given a string s containing only digits, Your task is to complete the function genIp which returns a vector containing all possible combination of valid IPv4 ip address and takes only a string s as its only argument .
Note : Order doesn't matter

For string 11211 the ip address possible are
1.1.2.11
1.1.21.1
1.12.1.1
11.2.1.1
"""

def genIP(digits):

    def valid_part(number):
        return 0 < len(number) <= 3 and (number[0] != '0' or number == "0") and 0 <= int(number) <= 255

    def backtrack(start, number):
        if start >= len(digits):
            return

        if number == 4:
            if valid_part(digits[start:]):
                yield digits[start:]
            return

        for i in range(1, 4):
            prefix = digits[start:start+i]
            if valid_part(prefix):
                for sol in backtrack(start + i, number + 1):
                    yield digits[start:start+i] + "." + sol

    return list(backtrack(start=0, number=1))


print(genIP("11211"))
print(genIP("1111"))
print(genIP("50361"))
print(genIP("67535629"))
print(genIP("237592"))
