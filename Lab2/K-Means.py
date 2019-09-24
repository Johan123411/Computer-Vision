from PIL import Image
import numpy as np
import random as rn
import math as mt


########################################
# value of K for k-means
#########################################


def main():
    input_img = Image.open("white-tower.png")
    input_img.show()
    temp_img = np.asarray(input_img)
    r, c, var = temp_img.shape
    iterations = 1
    convergance = 0
    centres = []
    while (len(centres) != 10):  # this is the value of K i.e. there are 10 centers and clusters
        randomC = [rn.randint(0, (r - 1)), rn.randint(0, (c - 1)),
                   temp_img[rn.randint(0, (r - 1))][rn.randint(0, (c - 1))][0],
                   temp_img[rn.randint(0, (r - 1))][rn.randint(0, (c - 1))][1],
                   temp_img[rn.randint(0, (r - 1))][rn.randint(0, (c - 1))][2]]
        if (randomC not in centres):
            centres.append(randomC)

    while convergance == 0:
        convergance = 1
        clusters = [[] for i in range(len(centres))]  # since there are 10 K means points, there will be 10
        # clusters
        for i in range(r):
            for j in range(c):
                distance = 9999.0
                for k in range(len(centres)):
                    Red = int(centres[k][2]) - int(temp_img[i][j][0])  # calculating the difference between the
                    # randomly generated centers Red Value and the Red Value of the given Image
                    Green = int(centres[k][3]) - int(temp_img[i][j][1])  # calculating the difference between the
                    # randomly generated centers Green Value and the Green Value of the given Image
                    Blue = int(centres[k][4]) - int(temp_img[i][j][2])  # #calculating the difference between the
                    # randomly generated centers RBlue Value and the Blue Value of the given Image
                    MeanVal = (int(centres[k][2]) + int(temp_img[i][j][0])) / 2
                    Color_Distance = float(mt.sqrt((((512 + MeanVal) * Red * Red) / 256) + (4 * Green * Green) + (
                            ((767 - MeanVal) * Blue * Blue) / 256)))
                    if (Color_Distance < distance):
                        distance = Color_Distance
                        Index = k;  # starts at index 0, then goes on till go on till k == 10
                clusters[Index].append([i, j, temp_img[i][j][0], temp_img[i][j][1], temp_img[i][j][2]])
        NewCentCount = 0

        for i in range(len(clusters)):
            point_count = len(clusters[i])
            print(point_count)
            red = 0
            green = 0
            blue = 0
            for points in clusters[i]:
                # this is because clusters = [[], [], []...] and inside [[x, y, r, g, b], [...], ... ]
                #                                                        0  1  2  3  4
                red = red + (int(points[2]) * int(points[2]))
                green = green + (int(points[3]) * int(points[3]))
                blue = blue + (int(points[4]) * int(points[4]))

            # if(point_count == 0):
            #     print("some issue with Random Numbers")
            #     exit()

            if (point_count != 0):
                red = int(mt.sqrt(red / point_count))
                green = int(mt.sqrt(green / point_count))
                blue = int(mt.sqrt(blue / point_count))

            # here we notice that the length of some clusters is 0 , in which case K-Means takes longer to execute
            new_bin = [red, green, blue]  # here we assign the new colours to the new color space
            old_bin = [centres[i][2], centres[i][3], centres[i][4]]  # this is the old color- space
            print("this is  BIN")
            print(new_bin)
            print(old_bin)

            # if(abs(new_bin[0]-old_bin[0])<=0.000025 and abs(new_bin[1]-old_bin[1])<=0.000025 and abs(new_bin[2]-old_bin[2])<=0.000025 ):
            if new_bin != old_bin:
                NewCentCount += 1
                centres[i][
                    2] = red  # in this case if the new color space is not equal to the older color space , then we iterate over the loop again
                centres[i][3] = green  # assign the new color centers
                centres[i][4] = blue
                convergance = 0  # this is what makes the flag condition fail

        print("Number of centers: ", NewCentCount)
        print()

    Imgae = np.asarray(
        Image.new('RGB', (c, r)))  # creating a new RGB image with the same dimentions as the original image.
    Imgae.setflags(write=1)
    for i in range(len(centres)):
        cent = centres[i]
        RGBval = [cent[2], cent[3], cent[4]]
        for j in clusters[i]:
            Imgae[j[0]][j[1]] = RGBval
    # here we write the new RGB value to the image, for K-means
    # this basically is used to reduce the size of the image, or compress the image. This is done by taking the major color centers k and relegating everything else
    FINAL_IMAGE = Image.fromarray(Imgae)
    FINAL_IMAGE.show()


main()
