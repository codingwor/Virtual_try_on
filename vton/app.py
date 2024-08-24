from flask import Flask, request, render_template
import base64
import numpy as np
import cv2

app = Flask(__name__)


def process_image(model_image, clothing_image):
    try:
        # Read the images using OpenCV
        model = cv2.imdecode(np.frombuffer(
            model_image, np.uint8), cv2.IMREAD_UNCHANGED)
        clothing = cv2.imdecode(np.frombuffer(
            clothing_image, np.uint8), cv2.IMREAD_UNCHANGED)

        # Convert the clothing image to HSV
        hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 52, 72])
        upper_green = np.array([102, 255, 255])

        # Create masks to isolate the green screen
        mask_white = cv2.inRange(hsv, lower_green, upper_green)
        mask_black = cv2.bitwise_not(mask_white)

        # Prepare the masks for bitwise operations
        mask_black_3CH = cv2.merge([mask_black, mask_black, mask_black])
        mask_white_3CH = cv2.merge([mask_white, mask_white, mask_white])

        # Apply the masks to the clothing image
        dst3 = cv2.bitwise_and(clothing, mask_black_3CH)

        # Resize the clothing image to match the model's dimensions
        model_h, model_w = model.shape[:2]
        clothing_resized = cv2.resize(dst3, (model_w, model_h))

        # Create a region of interest on the model image where the clothing will be placed
        roi = model[0:model_h, 0:model_w]

        # Create a mask of the clothing and its inverse mask
        clothing_gray = cv2.cvtColor(clothing_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(clothing_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the clothing in the ROI
        model_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only the region of the clothing from the clothing image
        clothing_fg = cv2.bitwise_and(
            clothing_resized, clothing_resized, mask=mask)

        # Put the clothing in the ROI and modify the main image
        dst = cv2.add(model_bg, clothing_fg)
        model[0:model_h, 0:model_w] = dst

        # Convert the final output to a format suitable for displaying in HTML
        _, buffer = cv2.imencode('.png', model)
        result_image = base64.b64encode(buffer).decode('ascii')

        return result_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            model_file = request.files['model'].read()
            clothing_file = request.files['clothing'].read()

            # Convert the images to base64 for displaying
            model_image = base64.b64encode(model_file).decode('ascii')
            clothing_image = base64.b64encode(clothing_file).decode('ascii')

            result_image = process_image(model_file, clothing_file)

            if result_image:
                return render_template('index.html', model_image=model_image, clothing_image=clothing_image, result_image=result_image)
            else:
                return "Error processing images", 500
        except Exception as e:
            print(f"Error handling file upload: {e}")
            return "Error handling file upload", 500
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
