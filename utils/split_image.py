from PIL import Image

# function to split a single image into half
def split_image(image_path, output_path1, output_path2):
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Initialize middle_x
    middle_x = 512

    # Split the image into two parts
    left_half = img.crop((0, 0, middle_x - 1, height))
    right_half = img.crop((middle_x + 1, 0, width, height))

    # Save the two parts
    left_half.save(output_path1)
    right_half.save(output_path2)


# call it
split_image('./images/300.png', './images/300-before.png', './images/300-after.png')
