import dataset as ds
from neural_network import NNetwork


def choose_action(num: int, network: NNetwork):
    print(num, ": ")
    data = None

    if num == 0:
        data = ds.BAD_ZERO
    elif num == 1:
        data = ds.BAD_ONE
    elif num == 2:
        data = ds.BAD_TWO
    elif num == 3:
        data = ds.BAD_THREE
    elif num == 4:
        data = ds.BAD_FOUR
    elif num == 5:
        data = ds.BAD_FIVE
    elif num == 6:
        data = ds.BAD_SIX
    elif num == 7:
        data = ds.BAD_SEVEN
    elif num == 8:
        data = ds.BAD_EIGHT
    else:
        data = ds.BAD_NINE

    print("-----------------------------------")
    network.recognize(data)


def main():
    network = NNetwork()

    while True:
        print("Input number [0-9] of [exit]: ")
        try:
            s = input()
            if s == 'exit':
                break
            num = int(s)
            if 0 <= num <= 9:
                choose_action(num, network)
        except ValueError:
            print("Incorrect value")


if '__main__' == __name__:
    main()
