import dataset as ds
from neural_network import NNetwork


def main():
    network = NNetwork()

    print("Zero: ")
    network.recognize(ds.ZERO)
    print("----------------------")
    print("Two: ")
    network.recognize(ds.TWO)
    print("----------------------")
    print("Seven: ")
    network.recognize(ds.SEVEN)
    print("----------------------")
    print("Five: ")
    network.recognize(ds.FIVE)
    print("----------------------")
    print("Nine: ")
    network.recognize(ds.NINE)


if '__main__' == __name__:
    main()
