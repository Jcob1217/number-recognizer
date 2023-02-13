from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plot
import pygame, sys
from pygame.locals import *
import tensorflow as tf

def train_model():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)
    model.save("model.h5")

    return model

# Changing list RGB values to grayscale
def rgb_to_grayscale(array):
    final_array = np.empty((0, 28))
    for row in array:
        _row =  row.flatten()[::3]
        final_array = np.append(final_array, np.array([_row]), axis=0)
    return final_array

# Resizing a given image to 28x28 
def resize_image():
    image = Image.open('image.jpeg')

    new_image = image.resize((28,28))
    new_image.save('image.jpeg')

    data = np.array(new_image)

    my_data = rgb_to_grayscale(data) / 255
    
    return my_data

# Creating a pygame window for drawing a number inside
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