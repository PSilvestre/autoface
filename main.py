import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import threading

from comm import Shutdown, EncodeRequest, DecodeRequest, EncodeResponse, DecodeResponse
from network import CNNAutoEncoder, I_WH
from ui import UI

from queue import Queue

BATCH_SIZE = 16
INITIAL_LEARNING_RATE = 5e-3
GAMMA = 0.9

NUM_EPOCHS = 50


def load_data():
    print("Begin loading data!")
    transform = transforms.Compose([
        transforms.Resize((I_WH, I_WH)),
        transforms.ToTensor(),

    ])
    data = datasets.ImageFolder("./dataset", transform)

    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print("Done loading data!")
    return data_loader


def train(data_loader, load_prev=False):
    print("Begin training!")
    # Load dataset
    # Probably using a DataLoader with ToTensor transform

    model = None
    if load_prev:
        model = load_trained_model()
    else:
        model = CNNAutoEncoder()
    print("Begin load model into GPU")
    model = model.cuda()
    print("Done load model into GPU")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        for (img, _) in data_loader:
            img = img.cuda()
            reencoded = model(img)

            loss = criterion(reencoded, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        plot_examples((img, reencoded), epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'model_weights_chk{}.pth'.format(epoch))
        print(f'Epoch:{epoch}, Loss: {loss.item():.4f}, LR: {INITIAL_LEARNING_RATE * GAMMA ** epoch:.6f}')
    # Forward
    # Backward
    # Save weights

    print("Done training!")
    torch.save(model.state_dict(), 'model_weights.pth')
    pass


def plot_examples(outputs, epoch):
    fig = plt.figure(figsize=(5, 2))
    imgs = outputs[0].detach().cpu().numpy()
    recon = outputs[1].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        if i >= 5: break
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.moveaxis(item, 0, -1))

    for i, item in enumerate(recon):
        if i >= 5: break
        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(np.moveaxis(item, 0, -1))
    plt.savefig("epoch_{}".format(epoch) + ".png")
    plt.close(fig)


def inference_server(ui_to_inf: Queue, inf_to_ui: Queue):
    model = load_trained_model()
    model.eval()

    r = ui_to_inf.get(block=True)
    while not isinstance(r, Shutdown):
        # process
        if isinstance(r, EncodeRequest):
            latent = model.encode(torch.tensor(r.input_img).requires_grad_(False).unsqueeze(0))
            inf_to_ui.put(EncodeResponse(latent=latent[0].detach().numpy()))
        if isinstance(r, DecodeRequest):
            output = model.decode(torch.tensor(r.latent).requires_grad_(False).unsqueeze(0))
            inf_to_ui.put(DecodeResponse(output_img=np.moveaxis(output[0].detach().numpy(), 0, -1)))

        # read next
        r = ui_to_inf.get()


def launch_inference_gui():
    ui_to_inf = Queue()
    inf_to_ui = Queue()

    inf_server_t = threading.Thread(target=inference_server, args=(ui_to_inf, inf_to_ui))
    inf_server_t.start()

    ui = UI(ui_to_inf, inf_to_ui)
    ui.launch()

    # pre-proc to correct size
    # encode
    # GUI Loop
    #    decode and display

    ui_to_inf.put(Shutdown())
    inf_server_t.join()


def load_trained_model():
    print("Begin loading model!")
    model = CNNAutoEncoder()
    model.load_state_dict(torch.load('model_weights.pth'))
    print("Done loading model!")
    return model


if __name__ == '__main__':
    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        print("CUDA not available. Exiting.")
        exit(1)

    dev_no = torch.cuda.current_device()
    print("Using device", torch.cuda.get_device_name(dev_no))

    #train(load_data(), False)
    launch_inference_gui()
