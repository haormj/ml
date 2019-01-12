if __name__ == '__main__':
    s = input()
    vector = s.split(' ')
    n = input()
    result = 0
    for i in vector:
        result += int(i) ** int(n)
    print(pow(result, 1/int(n)))
