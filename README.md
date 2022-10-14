# Knowledge Distilation [**(Paper)**](https://arxiv.org/pdf/2006.05525.pdf)

Knowledge distilation is a kind of transfer learning that learn from a larger pretrained model

The learning process of knowledge distillation is similar to the human beings learning

Students learn by watching and copying how teachers do it

If a teacher has a better ability, students will have a better ability too

On the contrary, if a teacher lacks ability, a student cannot produce good ability

We implemented response based offline distilation mentioned in the paper

The target training value of the student model is the predicted value of the teacher model

<img src="https://user-images.githubusercontent.com/43339281/195769566-a049cfa3-3923-4440-9854-3c563e69250f.png" width="500px">

It does not matter whether the model to be trained is a classification model, an object detection model, or an RNN model

The preprocessing logic to build the training tensor is also not required at all

All you need is a teacher model and a student model with the same input shape and output shape

<img src="https://user-images.githubusercontent.com/43339281/195769228-ad296d8f-bf8a-4470-af1b-0b4dfa277624.png" width="1000px">
