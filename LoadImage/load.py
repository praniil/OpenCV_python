import cv2

def loadImg():
    image = cv2.imread("./cricket.jpg")

    if image is None:
        print("Error: Couldnot open or find the image.")
        return
    
    cv2.imshow("Loaded Image", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()