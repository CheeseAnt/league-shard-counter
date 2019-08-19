import numpy as np
import cv2
import pandas as pd
import os

from time import time
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
        self.N_BOUNDS = np.array([70, 93, 70, 95]) / self.N_SCALE

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

        # confidence level of new champ or recognized
        self.CHAMP_CONFIDENCE = 0.79

        # one day for csv refresh time limit
        self.TIME_LIMIT = 86400

        # config for tesseract
        self.P_CONFIG = ("-l eng --oem 1 --psm 7")
        
        # pre-load the champion images
        self.champ_images = {champ : cv2.imread(champ) for champ in listdir("champs/*")}

        # keeps track of champ table update
        self.last_update = 0

    ## locate
    def assign_champ_to_square(self, imagelist, display=False):
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
            base_ar = Counter(areas).most_common()[0][0]
            
            # also look at similar areas
            freqs = {x for x in areas if abs(base_ar - x) < base_ar / 20}

            # collect the actual image slices
            imboxes = list()
            for ar in freqs:
                for idx, bbox in enumerate(boxes[ar]):
                    imboxes.append(base[bbox[0]:bbox[0] + bbox[2],
                        bbox[1]:bbox[1] + bbox[3]])

            # calculate the correlations between each of the champions
            corrs = self.list_corr(imboxes)

            # look through rates and see if any new champions have been found
            for idx, (rate, champ, box) in enumerate(corrs):
                if display:
                    print("Found {} at {}".format(champ, rate))
                if rate < self.CHAMP_CONFIDENCE:
                    print("New Champion Found")
                    cv2.imwrite("champs/champ_" + str(idx).zfill(3) + ".png", box)

                # if the champion is not in the last, add em
                if champ not in all_champs:
                    all_champs[champ] =  (rate, champ, box)

            # info
            print("Found {} champions".format(len(corrs)))
        
        return list(all_champs.values())

    ## produce a list of correlations of each bbox against known champions
    def list_corr(self, boxes):
        champ_corrs = list()

        # for each bbox, compute relative correlation
        for box in boxes:
            max_match = 0
            c_champ = None

            # compare against every known champion
            for champ in self.champ_images:
                match = self.correlate(box, self.champ_images[champ])

                # overwrite if higher confidence
                if match > max_match:
                    max_match = match
                    c_champ = champ

            # append highest match
            champ_corrs.append((max_match, c_champ, box))

        return champ_corrs

    ## find correlation of base image against a comparison
    def correlate(self, base, comp):
        # resize the larger image
        if base.shape[0] > comp.shape[0]:
            base = cv2.resize(base, (comp.shape[1], comp.shape[0]))
        else:
            comp = cv2.resize(comp, (base.shape[1], base.shape[0]))

        # compute normalized correlation
        results = cv2.matchTemplate(base, comp, cv2.TM_CCORR_NORMED)

        return results.max()

    ##  get datasheet of current champions and their essence costs
    def getChampTable(self):
        if (time() - self.last_update) > self.TIME_LIMIT:
            # obtain  the html
            res = geturl(self.CHAMP_URL)    
            soup = bs(res.content, 'lxml')

            # find the table and convert to dataframe
            table = soup.find_all("table", {"class" : "sortable"})[0]
            self.champ_table = pd.read_html(str(table))[0]
            
            # sift through and extract just the name
            for idx, row in self.champ_table.iterrows():
                name = row["Champion"].replace(",", "\xa0").split("\xa0")[0]

                self.champ_table.at[idx, "Champion"] = name

        return self.champ_table

    ## convert a champion from pathname into proper name and blue essence cost
    def convChamp(self, champ, cdata):
        # find the basename of the champion
        champ = os.path.splitext(os.path.basename(champ))[0]

        # find corresponding champion in champ table
        row = cdata[cdata["Champion"].str.lower() == champ]

        # convert to cost
        cost = int(self.EXCHANGE[row["Blue Essence"].values[0]])
        name = row["Champion"].values[0]

        return cost, name

    ## get the bounding box of the number
    def getNumberBox(self, img):
        boundsy = np.int32(self.N_BOUNDS[:2] * img.shape[1])
        boundsx = np.int32(self.N_BOUNDS[2:] * img.shape[0])
        return img[boundsy[0]:boundsy[1], boundsx[0]:boundsx[1]]

    ## retrieve the digit as text from the image
    def getDigit(self, image):
        # convert to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh on anything over 127, inverted for OCR as similar to MNIST - black on white bg
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # get the string from pytesseract
        result = image_to_string(image, config=self.P_CONFIG)
        # search for digit 
        groups = search("\d+", result)
        
        # if digit found use it, else assume one
        if groups:
            result = int(groups.group())
        else:
            result = 1

        return result

    ## count the shards in the given set of images
    def countShardCosts(self, images):
        corrs = sc.assign_champ_to_square(images, display=False)

        total_cost = 0
        total_shards = 0

        df = pd.DataFrame()

        # update known champ data
        self.getChampTable()

        # sort the list based on the champion name
        corrs.sort(key=lambda x : x[1])
        for rate, champ, box in corrs: 
            # find the champion name and cost
            cost, name = self.convChamp(champ, self.champ_table)
            # retrieve number of shards via OCR
            num_shards = self.getDigit(self.getNumberBox(box))

            # calculate cost
            n_cost = cost * num_shards
            total_shards += num_shards
            total_cost += n_cost

            # print result
            print("{} champion shards of {} is worth {} blue essence".format(
                num_shards, name, n_cost))

            df = df.append({"Champion" : name, "Cost": cost, "Shards" : num_shards,
                "Total Cost" : n_cost}, ignore_index=True)
      
        print("Total cost of all champion shards are {} blue essence".format(
            total_cost))
        print("{} shards found in total from {} unique champions".format(total_shards,
            len(corrs)))

        return df

    ## count shards using a list of paths rather than images
    def countShardsPath(self, imagepaths):
        images = [cv2.imread(image) for image in imagepaths]
        return self.countShardCosts(images)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("base", help="Base Image", nargs="+")

    args = parser.parse_args()

    sc = ShardCounter()

    sc.countShardsPath(args.base)
