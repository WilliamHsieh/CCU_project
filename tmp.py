x = [1, 2]

def haha():
    global x
    print(x)
    x.append(3)
    print(x)

haha()
print(x)

