import matplotlib.pyplot as pyplot
import numpy as np
from load_dataset_NPZ import load_dataset

print('Kill all humans!')

# получаем данные из архива
images, labels = load_dataset()

# рандомные веса: от, до, (строк, столбцов)
# 20 - hidden layer size
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
# print(weights_hidden_to_output)

# рандомные нейроны смещения (нули в столбик)
bias_input_to_hidden = np.zeros((20,1))
bias_hidden_to_output = np.zeros((10,1))
# print(bias_input_to_hidden)

# обучение = коррекция весов
epochs = 3 # learning iterations
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f'Epoch No: {epoch}')

    for image, label in zip(images, labels):
        #each element of the original matrix becomes a separate row in the new matrix.
        # The -1 in the reshape function is used as a placeholder
        # to automatically compute the size of that dimension based on the total number of elements,
        # ensuring that you get a column vector.

        ##here will be column of image 784 pixels values for input
        image = np.reshape(image, (-1, 1))

        #here will be column of label (index) (5 = 0000010000) for output
        label = np.reshape(label, (-1, 1))

        ### 1 step forward propagation
        # 1.1. produse (multiplicate) image on weights and add bias
        #20x784 @ 784x1 = 20x1     + 20x1
        # print(f'weights_input_to_hidden.shape {weights_input_to_hidden.shape}')
        # print(f'image.shape {image.shape}')
        # print(f'bias_input_to_hidden.ndim {bias_input_to_hidden.ndim}')
        # hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden
        hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden

        ## 1.2. normalizing = activation (Linear or ReLU or Sigmoid)
        #Linear                 f(x) = x
        #Rectified Linear Unit  f(x) = np.maximum(0, matrix)    replaces all negative values in the input
        #       matrix with zero while leaving positive values unchanged
        #Sigmoid                f(x) = 1/(1+e**-x)
        hidden = 1/(1 + np.exp(-hidden_raw)) #sigmoid

        # 1.3. forward propagation to outpot
        # = 10x20 @ 20x1    = 10x1     + 10x1
        # print(f'weights_hidden_to_output.shape {weights_hidden_to_output.shape}')
        # print(f'hidden.shape {hidden.shape}')
        # print(f'bias_hidden_to_output.shape {bias_hidden_to_output.shape}')
        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        # print(output_raw)

        # 1.4. normalizing output
        # output = np.maximum(0, output_raw)
        output = 1 / (1 + np.exp(-output_raw))  # sigmoid
        # print(f'output = {output}')
        # print(f'label = {label}')

        # 1.4. Loss|Err calculation

        # Mean Squared Error method = 1/N*(сумма квадратов отклонений `Yi - Yi` для i из N)
        e_loss += 1/len(output)*np.sum((output-label)**2, axis=0) #число
        # print(f'e_loss {e_loss}')

        # is index of max value in label == index of max value in output
        # одинаковы ли индексы максимумов в ответет и эталоне
        e_correct += int(np.argmax(output) == np.argmax(label))  #булево 1 или 0
        # print(f'np.argmax(output) {np.argmax(output)}')
        # print(f'np.argmax(label) {np.argmax(label)}')
        # print(f'e_correct {e_correct}')

        # 2.BackPropagation
        # 2.1. output layer
        # разница двух массвов размером 10х1 = 10х1
        delta_output = output-label
        # 10х20 += -число * 10х1 @ 1*20
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        # 10x1 += -число * 10х1
        bias_hidden_to_output += -learning_rate * delta_output

        # 2.2. hidden layer
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

        # break

    print(f'Loss: {round((e_loss[0]/images.shape[0]) * 100, 3)} %')
    print(f'Accuracy: {round((e_correct / images.shape[0]) * 100, 3)} %')
    e_loss = 0
    e_correct = 0
    # break