import cv2


def plot_curve_regions(image_path, contact, adhesion, output_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224,224))

    height = image.shape[0]

    cv2.line(image, (contact,0), (contact,height), (0,255,0), 2)

    cv2.line(image, (adhesion,0), (adhesion,height), (0,0,255), 2)

    cv2.putText(image, "contact", (contact+5,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

    cv2.putText(image, "adhesion", (adhesion+5,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

    cv2.imwrite(output_path, image)