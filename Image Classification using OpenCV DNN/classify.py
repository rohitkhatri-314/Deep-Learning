import cv2 as cv

# Load model

weights="mobilenet.caffemodel"
architecture="mobilenet_deploy.prototxt"
net=cv.dnn.readNetFromCaffe(architecture,weights)

# Load image
image = cv.imread('image2.png')
blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224),
                            mean=(104, 117, 123))  # Mean values from Caffe training

# Set input and perform forward pass
net.setInput(blob)
output = net.forward()

# Load class labels
with open("synset.txt") as f:
    classes = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Get top prediction
class_id = output[0].argmax()
confidence = output[0][class_id]

# Show result
label = f"{classes[class_id]}: {float(confidence)*100:.2f}%"
print("Prediction:", label)

# Display
cv.putText(image, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.imshow("Classification", image)
cv.waitKey(0)
cv.destroyAllWindows()
