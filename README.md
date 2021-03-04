# imagination

AI based `lane count` / `road types` feature extraction from aerial imagery in switzerland.

## Dataset

The [swissimage](https://shop.swisstopo.admin.ch/en/products/images/ortho_images/SWISSIMAGE10) dataset cconsists of cloud free areal imagery with a ground resolution of 10 centimeters.

AI based feature extraction from aerial imagery has shown promising results in several studies for road features (eg Mnhi) . Various implementations are available eg roadtagger or airs

## Classify road types

There exist various [road type calssifications](https://de.wikipedia.org/wiki/Strassensystem_in_der_Schweiz_und_in_Liechtenstein#Schweiz_2) in switzerland which we may try to fit.

### Supervised way

Train a classifier, DenseNet or ResNet. Maybe DenseNet is better since there is a model for DenseNet in tensorflow 2.0, one can just call that class and use.

### Unsupervised way

In case there is no any label in the dataset, to find out the road type, we can manually crop images of different types of road and use these images as reference. Then we can compute the perceptual loss between the image and the reference. The smallest loss yields the type/class of the input image. Perceptual loss can be easily computed by the pretrained model [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16). Unlike pixel-wise loss such as L2 loss, perceptual loss focuses on the content of the image, more tolerant of the translation, rotation of the objects.

## Count lanes

To count the marks or edges of road lanes, edge detection might be efficient. If the backgrounds/surroundings are noisy, we can use bileteral filter to preprocess the image.

## Resources

- https://github.com/robmarkcole/satellite-image-deep-learning
- https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
- https://github.com/mitroadmaps/roadtagger
- https://github.com/mahmoudmohsen213/airs
