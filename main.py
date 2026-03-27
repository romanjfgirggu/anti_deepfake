from traning import Model

def main():
    model = Model()
    m  = model.traning_model()
    model.save_model(model=m)


if __name__ == '__main__':
    main()