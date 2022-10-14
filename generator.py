"""
Authors : inzapp

Github url : https://github.com/inzapp/knowledge-distilation

Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from util import ModelUtil


class DataGenerator:
    def __init__(self, teacher_model, image_paths, input_shape, batch_size):
        self.generator_flow = GeneratorFlow(teacher_model, image_paths, input_shape, batch_size)

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def flow(self):
        return self.generator_flow


class GeneratorFlow:
    def __init__(self, teacher_model, image_paths, input_shape, batch_size):
        self.teacher_model = teacher_model
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_width, self.input_height, self.input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        self.batch_size = batch_size
        self.batch_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.device = ModelUtil.check_available_device()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
            
    def __getitem__(self, index):
        fs, batch_x, batch_y = [], [], []
        for path in self.get_next_batch_image_paths():
            fs.append(self.pool.submit(ModelUtil.load_img, path, self.input_channel))
        for f in fs:
            img, _, _ = f.result()
            img = ModelUtil.resize(img, (self.input_width, self.input_height))
            x = ModelUtil.preprocess(img)
            batch_x.append(x)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(ModelUtil.graph_forward(self.teacher_model, batch_x, self.device)).astype('float32')
        return batch_x, batch_y

    def get_next_batch_image_paths(self):
        start_index = self.batch_size * self.batch_index
        end_index = start_index + self.batch_size
        batch_image_paths = self.image_paths[start_index:end_index]
        self.batch_index += 1
        if self.batch_index == self.__len__():
            self.batch_index = 0
            np.random.shuffle(self.image_paths)
        return batch_image_paths

