

def eval(tokens: str) -> int:
    operators = []
    operands = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            lo = i
            while tokens[i].isdigit():
                i += 1
            operands.append(int(tokens[lo:i]))
        else:
            if token == ')':
                op = operators.pop()
                arg2 = operands.pop()
                arg1 = operands.pop()
                operands.append(op(arg1, arg2))
            elif token == '+':
                operators.append(lambda x, y: x + y)
            elif token == '*':
                operators.append(lambda x, y: x * y)
            elif token == '-':
                operators.append(lambda x, y: x - y)
            elif token == '/':
                operators.append(lambda x, y: x / y)
            i += 1

    return operands.pop()


print(eval("((1+2)*(12-5))"))
