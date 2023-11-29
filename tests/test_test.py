import training

def test_training():
    print(training.test())
    assert training.test() == "It works!"

if __name__ == '__main__':
    test_training()
