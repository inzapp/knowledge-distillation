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
import os
import tensorflow as tf

from util import ModelUtil
from generator import DataGenerator
from lr_scheduler import LRScheduler


class KnowledgeDistilation:
    def __init__(self,
                 teacher_model_path='./teacher_model.h5',
                 student_model_path='./student_model.h5',
                 train_image_path='',
                 lr_policy='onecycle',
                 lr=0.01,
                 momentum=0.9,
                 burn_in=1000,
                 batch_size=32,
                 iterations=100000,
                 checkpoints='checkpoints'):
        self.lr = lr
        self.momentum = momentum
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr_policy = lr_policy
        self.checkpoints = checkpoints

        if os.path.exists(teacher_model_path) and os.path.isfile(teacher_model_path):
            self.teacher_model = tf.keras.models.load_model(teacher_model_path)
        else:
            print(f'teacher model not found : {teacher_model_path}')
            exit(0)

        if os.path.exists(student_model_path) and os.path.isfile(student_model_path):
            self.student_model = tf.keras.models.load_model(student_model_path)
        else:
            print(f'student model not found : {student_model_path}')
            exit(0)

        teacher_model_input_shape = self.teacher_model.input_shape[1:]
        student_model_input_shape = self.student_model.input_shape[1:]
        if teacher_model_input_shape != student_model_input_shape:
            print('teacher and student model input shape is differ')
            print('teacher model input shape : {teacher_model_input_shape}')
            print('student model input shape : {student_model_input_shape}')
            exit(0)

        teacher_model_output_shape = self.teacher_model.output_shape[1:]
        student_model_output_shape = self.student_model.output_shape[1:]
        if teacher_model_output_shape != student_model_output_shape:
            print('teacher and student model output shape is differ')
            print('teacher model output shape : {teacher_model_output_shape}')
            print('student model output shape : {student_model_output_shape}')
            exit(0)

        ModelUtil.set_channel_order(teacher_model_input_shape)
        self.train_image_paths = ModelUtil.init_image_paths(train_image_path)

        self.train_data_generator = DataGenerator(
            teacher_model=self.teacher_model,
            image_paths=self.train_image_paths,
            input_shape=teacher_model_input_shape,
            batch_size=batch_size)

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
            loss_mean = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_mean

    def fit(self):
        os.makedirs(f'{self.checkpoints}', exist_ok=True)
        print('\nteacher model summary')
        self.teacher_model.summary()
        print('\nstudent model summary')
        self.student_model.summary()
        print(f'\nteacher model input shape : {self.teacher_model.input_shape}')
        print(f'student model input shape : {self.student_model.input_shape}')
        print(f'teacher model output shape : {self.teacher_model.output_shape}')
        print(f'student model output shape : {self.student_model.output_shape}')
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('\nstart training')
        iteration_count = 0
        optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum)
        lr_scheduler = LRScheduler(iterations=self.iterations, lr=self.lr)
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                lr_scheduler.update(optimizer, iteration_count, self.burn_in, self.lr_policy)
                loss = self.compute_gradient(self.student_model, optimizer, batch_x, batch_y)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.9f}', end='')
                if iteration_count % 1000 == 0:
                    self.student_model.save(f'{self.checkpoints}/model_{iteration_count}_iter.h5', include_optimizer=False)
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    return

