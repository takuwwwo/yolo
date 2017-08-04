import tensorflow as tf
import network
import yolo_input

def main(_):
    # The image size of input image
    image_size = network.IMAGE_SIZE

    # The number of class.
    C = network.C

    # The last output image size
    S = network.S

    # The number of obejects which can be detected in a cell
    B = network.B

    # The depth of last layer.
    last_output = network.last_output

    # batch size
    batch_size = yolo_input.batch_size

    # Create placeholders for import
    images = tf.placeholder(tf.float32, (batch_size, image_size, image_size, 1))
    labels = tf.placeholder(tf.float32, (batch_size, image_size, image_size, last_output))
    obj = tf.placeholder(tf.float32, (batch_size, image_size, image_size))
    noobj = tf.placeholder(tf.float32, (batch_size, image_size, image_size))

    # Build a graph that computes the logits predictions from the inference model.
    logits = network.inference(images)

    # Calculate loss
    loss = network.loss(logits, labels, obj, noobj)

    # Build a graph that trains the model with one batch of examples and updates the model parameters
    train_op = network.training(loss, 0.01)

    """
    up to here, build a model, not start training.
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            i, l, ob, noob = yolo_input.nextbatch(batch_size)  #TODO
            if i % 10 == 9:
                curr_loss = sess.run(loss, {images: i, labels: l, obj: ob, noobj: noob})
                print('step: %d, loss: %f' %(i, curr_loss))
            train_op.run(feed_dict={images: i, labels: l, obj: ob, noobj: noob})

if __name__ == '__main__':
    tf.app.run(main=main)


