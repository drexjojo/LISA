import torch
FILE = "../data/trained_models/low_resource_japanese_music_lang_disc.chkpt"

def main():
    trained_model = torch.load(FILE)
    print(trained_model["acc"])


if __name__=="__main__":
    main()