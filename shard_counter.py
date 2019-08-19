import numpy as np
import cv2
import pandas as pd
import os

from pytesseract import image_to_string
from re import search
from glob import glob as listdir
from collections import Counter
from argparse import ArgumentParser
from bs4 import BeautifulSoup as bs
from requests import get as geturl

class ShardCounter():
    def __init__(self):
        # measured values from own data
        self.N_SCALE = 98
        self.N_BOUNDS = np.array([70, 93, 70, 95]) / N_SCALE

        # url for table of champions from wiki
        self.CHAMP_URL = "https://leagueoflegends.fandom.com/wiki/List_of_champions"

        # blue essence exchange rates
        self.EXCHANGE = {
            450 : 90,
            1350 : 270,
            3150 : 630,
            4800 : 960,
            6300 : 1260,
            7800 : 1560
        }

        # instead of 256 colours have only 64
        self.IM_QUANT_FACTOR = 4

        # bounds on boundary thresholding
        self.LOWER_B_THRESH = np.array([30, 22, 18])
        self.UPPER_B_THRESH = np.array([80, 72, 48])

        # limit on the area of a champion square
        self.SQ_AREA_LIMIT = 1500

        # one day for csv refresh time limit
        self.TIME_LIMIT = 86400

        # config for tesseract
        self.P_CONFIG = ("-l eng --oem 1 --psm 7")
        
        # pre-load the champion images
        self.champ_images = {champ : cv2.imread(champ) for champ in listdir("champs/*")}

    def find_squares(self, imagelist, display=False):
        all_champs = dict()

        # iterate through images
        for base in imagelist:
            # quantize the image slightly
            smoothed = base // self.IM_QUANT_FACTOR
            smoothed = smoothed * self.IM_QUANT_FACTOR

            # threshold based on RGB values, more accurate than just BW
            rgbthresh = cv2.inRange(smoothed, self.LOWER_B_THRESH,
                self.UPPER_B_THRESH)
            rgbthresh = cv2.filter2D(rgbthresh, -1, np.ones((2, 2))) 

            # find contours within the images, essentially shape finding
            contours, h = cv2.findContours(rgbthresh, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            # holds bounding boxes and frequency of areas
            boxes = dict()
            areas = list()

            #  iterate through contours
            for cnt in contours:
                # calculate the approximate shape
                arclength = 0.1 * cv2.arcLength(cnt, True)
                shape = cv2.approxPolyDP(cnt,arclength,True)

                # if the shape has four points, i.e. a square
                if(len(shape) == 4):
                    # obtain the bounding rectangle 
                    (x, y, w, h) = cv2.boundingRect(shape)

                    # calculate the area
                    ar = w * h

                    # presume the area is above a limit
                    if ar > self.SQ_AREA_LIMIT:
                        # if not present in dictionary, add an entry
                        if ar not in boxes:
                            boxes[ar] = list()

                        # keep track of areas and bounding boxes
                        areas.append(ar)
                        boxes[ar].append([y, x, w, h])

            # find the most common area
            base_ar = Counter(areas).most_common[0][0]
            
            # also look at similar areas
            freqs = {x for x in areas if abs(base_ar - x) < base_ar / 20}

            # collect the actual image slices
            imboxes = list()
            for ar in freqs:
                for idx, bbox in enumerate(boxes[ar]):
                    imboxes.append(base[bbox[0]:bbox[0] + bbox[2],
                        bbox[1]:bbox[1] + bbox[3]])

            # calculate the correlations between each of the champions
            corrs = self.list_corr(imboxes, "champs/")

            for idx, (rate, champ, box) in enumerate(corrs):
                if display:
                    print("Found {} at {}".format(champ, rate))
                if rate < 0.79:
                    print("New Champion Found")
                    cv2.imwrite("champs/champ_" + str(idx).zfill(3) + ".png", box)

                if champ not in all_champs:
                    all_champs[champ] =  (rate, champ, box)

            print("Found {} champions".format(len(corrs)))
        
        return list(all_champs.values())

    def correlate(self, base, comp):
        if base.shape[0] > comp.shape[0]:
            base = cv2.resize(base, (comp.shape[1], comp.shape[0]))
        else:
            comp = cv2.resize(comp, (base.shape[1], base.shape[0]))

        results = cv2.matchTemplate(base, comp, cv2.TM_CCORR_NORMED)

        max_r = results.max()
        
        return max_r

    def list_corr(self, boxes, path):
        
        champ_corrs = list()

        for box in boxes:
            max_match = 0
            c_champ = None

            for champ in champ_images:
                match = correlate(box, champ_images[champ])

                if match > max_match:
                    max_match = match
                    c_champ = champ

            champ_corrs.append((max_match, c_champ, box))

        return champ_corrs

    

        
    def getData(self):
        res = geturl(self.CHAMP_URL)    
        soup = bs(res.content, 'lxml')
        table = soup.find_all("table", {"class" : "sortable"})[0]
        df = pd.read_html(str(table))[0]
        
        for idx, row in df.iterrows():
            name = row["Champion"].replace(",", "\xa0").split("\xa0")[0]

            df.at[idx, "Champion"] = name

        return df

    def convChamp(self, champ, cdata):
        champ = os.path.splitext(os.path.basename(champ))[0]
        row = cdata[cdata["Champion"].str.lower() == champ]

        cost = int(EXCHANGE[row["Blue Essence"].values[0]])
        name = row["Champion"].values[0]
        return cost, name

    def getNumberBox(self, img):
        boundsy = np.int32(N_BOUNDS[:2] * img.shape[1])
        boundsx = np.int32(N_BOUNDS[2:] * img.shape[0])
        return img[boundsy[0]:boundsy[1], boundsx[0]:boundsx[1]]

    def getDigit(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        result = image_to_string(image, config=config)
        groups = search("\d+", result)
        
        if groups:
            result = int(groups.group())
        else:
            result = 1

        return result

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("base", help="Base Image", nargs="+")

    args = parser.parse_args()

    corrs = find_squares(args.base, display=False)

    champ_data = pd.read_csv("champdata.csv")
    #getData().to_csv("champdata.csv")

    total_cost = 0
    total_shards = 0

    corrs.sort(key=lambda x : x[1])
    for rate, champ, box in corrs: 
        cost, name = convChamp(champ, champ_data)
        num_shards = getDigit(getNumberBox(box))
        cost = cost * num_shards
        total_shards += num_shards
        total_cost += cost
        print("{} champion shards of {} is worth {} blue essence".format(
            num_shards, name, cost))
  
    print("Total cost of all champion shards are {} blue essence".format(
        total_cost))
    print("{} shards found in total from {} unique champions".format(total_shards,
        len(corrs)))

