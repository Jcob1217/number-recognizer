from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plot
import pygame, sys
from pygame.locals import *
import tensorflow as tf

# Defining a function to train the model
def train_model():
    # Loading the MNIST dataset from TensorFlow Keras
    mnist = tf.keras.datasets.mnist

    # Splitting the dataset into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing the pixel values of the images in the dataset
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Creating the neural network model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())  # Flattening the input layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # Adding a hidden layer with 128 neurons and ReLU activation
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # Adding another hidden layer with 128 neurons and ReLU activation
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Adding an output layer with 10 neurons (one for each digit) and softmax activation

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model on the training set for 3 epochs
    model.fit(x_train, y_train, epochs=3)

    # Saving the trained model to a file
    model.save("model.h5")

    return model

# Defining a function to convert an RGB image to grayscale
def rgb_to_grayscale(array):
    final_array = np.empty((0, 28))
    for row in array:
        _row =  row.flatten()[::3]
        final_array = np.append(final_array, np.array([_row]), axis=0)
    return final_array

# Defining a function to resize an image to 28x28 pixels
def resize_image():
    # Opening an image file using the PIL library
    image = Image.open('image.jpeg')

    # Resizing the image to 28x28 pixels
    new_image = image.resize((28,28))
    new_image.save('image.jpeg')

    # Converting the image to a NumPy array
    data = np.array(new_image)

    # Converting the image to grayscale and normalizing its pixel values
    my_data = rgb_to_grayscale(data) / 255
    
    return my_data

# Defining a function to create a pygame window for drawing a number
def draw_number():
    pygame.init()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    mouse_position = (0, 0)
    drawing = False
    pygame.display.set_caption('Draw a number')
    screen = pygame.display.set_mode((280, 280), 0, 32)
    screen.fill(BLACK)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                # Saving the drawn image to a file and exiting the loop
                picture = pygame.image.save(screen, "image.jpeg")
                running = False
            elif event.type == MOUSEMOTION:
                if (drawing):
                    mouse_position = pygame.mouse.get_pos()
                    pygame.draw.rect(screen, WHITE, pygame.Rect(mouse_position[0], mouse_position[1], 5, 5))
            elif event.type == MOUSEBUTTONUP:
                drawing = False
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True

        pygame.display.update()
    
    pygame.quit()

def main():
    draw_number()
    picture = resize_image()

    model = train_model()

    saved_model = tf.keras.models.load_model("model.h5")

    prediction = saved_model.predict(np.array([picture]))

    plot.imshow(picture, cmap=plot.cm.binary)
    plot.title(f"Predicted: {np.argmax(prediction)}")
    plot.show()

if __name__ == "__main__":
    main()