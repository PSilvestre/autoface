import time

import PySimpleGUI as sg
import numpy as np

from comm import EncodeRequest, EncodeResponse, DecodeResponse, DecodeRequest
from network import LATENT_SPACE, I_WH

import matplotlib.pyplot as plt
from matplotlib import image as pltimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue
import tkinter as tk
from PIL import Image as im
from PIL import ImageTk as imtk
# def make_dpi_aware():
#  import ctypes
#  import platform
#  if int(platform.release()) >= 8:
#    ctypes.windll.shcore.SetProcessDpiAwareness(True)
OUTPUT_IMAGE_KEY = "-OUTPUT-IMAGE-"
SLIDER_KEY = "-SLIDER-"
INPUT_IMAGE_KEY = "-INPUT-IMAGE-"
FILE_KEY = "-FILE-"
RESET_BUTTON_KEY = "-RESET-"

DECODE_UPDATE_MILLIS = 500

milli_time = lambda: time.time_ns() // 1_000_000


class UI:

    def __init__(self, ui_to_inf: Queue, inf_to_ui: Queue):
        self.img_id = None
        self.ui_to_inf = ui_to_inf
        self.inf_to_ui = inf_to_ui

        self.title = "Pedros weekend autoencoder thingy"

        # make_dpi_aware()

        column_1 = [
            [
                sg.Text("Select Input Image"),
                sg.In(size=(25, 1), enable_events=True, key=FILE_KEY),
                sg.FileBrowse(),
            ],
            [
                sg.Text("Input Image Preview"),
                sg.Image(key=INPUT_IMAGE_KEY)
            ],
            [
                sg.Text("Reset to Original Image Values"),
                sg.Button(button_text="RESET", key=RESET_BUTTON_KEY, enable_events=True)
            ]
        ]

        # Column 2 contains sliders

        column_2 = []
        for i in range(int(LATENT_SPACE / 2)):
            column_2.append([sg.In(size=(10, 1), default_text="" + str(i)),
                             sg.Slider(range=(-1.0, 1.0), orientation="h", default_value=0.0, resolution=0.01,
                                       key=SLIDER_KEY + str(i), enable_events=True)])

        column_3 = []
        for i in range(int(LATENT_SPACE / 2), LATENT_SPACE):
            column_3.append([sg.In(size=(10, 1), default_text="" + str(i)),
                             sg.Slider(range=(-1.0, 1.0), orientation="h", default_value=0.0, resolution=0.01,
                                       key=SLIDER_KEY + str(i), enable_events=True)])
        # Column 3 displays output image
        column_4 = [[
            sg.Text("Reconstruction"),
            sg.Graph(key=OUTPUT_IMAGE_KEY, background_color="black", canvas_size=(I_WH, I_WH), graph_bottom_left=(0, 0),
                     graph_top_right=(I_WH, I_WH))
        ]
        ]

        self.layout = [[sg.Column(column_1), sg.Column(column_2), sg.Column(column_3), sg.Column(column_4)]]

    def draw_img(self, window, img_in: np.ndarray):

        img = (img_in * 255).astype(np.uint8)
        # turn our ndarray into a bytesarray of PPM image by adding a simple header:
        # this header is good for RGB. for monochrome, use P5 (look for PPM docs)

        # turn that bytesarray into a PhotoImage object:
        image = im.fromarray(img)
        #image = tk.PhotoImage(width=I_WH, height=I_WH, data=image)
        image = imtk.PhotoImage(image=image)

        # for first time, create and attach an image object into the canvas of our sg.Graph:
        if self.img_id is None:
            # I believe Widget is set to None here, which is why this fails
            self.img_id = window[OUTPUT_IMAGE_KEY].Widget.create_image((0, 0), image=image, anchor=tk.NW)
            #self.img_id = window[OUTPUT_IMAGE_KEY].CreateImage(data=image, location=(0,0))
            # we must mimic the way sg.Graph keeps a track of its added objects:
            window[OUTPUT_IMAGE_KEY].Images[self.img_id] = image
        else:
            # we reuse the image object, only changing its content
            window[OUTPUT_IMAGE_KEY].Widget.itemconfig(self.img_id, image=image)
            # we must update this reference too:
            window[OUTPUT_IMAGE_KEY].Images[self.img_id] = image

    def launch(self):
        window = sg.Window(self.title, self.layout, finalize=True)

        latent = np.zeros((LATENT_SPACE,))
        current_file = ""
        last_decode_update = milli_time()
        # fig_agg = self.draw_figure_in_canvas(window["CANVAS"].TKCanvas, fig)
        while True:
            event, values = window.read(timeout=125)

            if event == sg.WIN_CLOSED:
                break
            # if input img selected then send encode request
            if event == FILE_KEY and values[FILE_KEY] != current_file:
                current_file = values[FILE_KEY]
                self.request_encode(current_file, window)
            if event == RESET_BUTTON_KEY:
                self.request_encode(current_file, window)

            # if slider event, send decode request
            if SLIDER_KEY in event:
                time = milli_time()
                i = int(event[-1])
                latent[i] = values[event]
                if (time - last_decode_update) > DECODE_UPDATE_MILLIS:
                    self.ui_to_inf.put(DecodeRequest(latent=latent))
                    last_decode_update = time

            # while queue has data
            try:
                while True:
                    inf_response = self.inf_to_ui.get(block=False)
                    #   if encode respose then set sliders
                    if isinstance(inf_response, EncodeResponse):
                        latent = inf_response.latent
                        for i, val in enumerate(inf_response.latent):
                            window[SLIDER_KEY + str(i)].update(value=float(val))
                        self.ui_to_inf.put(DecodeRequest(latent=latent))

                    #   if decode response then update output img
                    if isinstance(inf_response, DecodeResponse):
                        self.draw_img(window, inf_response.output_img)
            except:
                pass

        window.close()

    def request_encode(self, file_path, window):
        image = pltimg.imread(file_path)
        # TODO format image to correct size
        self.ui_to_inf.put(EncodeRequest(input_img=np.moveaxis(image, -1, 0)))
        window[INPUT_IMAGE_KEY].update(file_path)
