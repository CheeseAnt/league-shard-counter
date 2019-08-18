import numpy as np
import cv2
import pandas as pd
import os

from pytesseract import image_to_string
from re import search
from glob import glob as listdir
from collections import Counter
from argparse import ArgumentParser
from shapely.geometry import Polygon
from bs4 import BeautifulSoup as bs
from requests import get as geturl

N_SCALE = 98
N_BOUNDS = np.array([70, 93, 70, 95]) / N_SCALE
config = ("-l eng --oem 1 --psm 7")

def correlate(base, comp):
    if base.shape[0] > comp.shape[0]:
        base = cv2.resize(base, (comp.shape[1], comp.shape[0]))
    else:
        comp = cv2.resize(comp, (base.shape[1], base.shape[0]))

    results = cv2.matchTemplate(base, comp, cv2.TM_CCORR_NORMED)

    max_r = results.max()
    
    return max_r

def list_corr(boxes, path):
    champs = listdir(path + "*")
    
    champ_images = {champ : cv2.imread(champ) for champ in champs}
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

def find_squares(imagelist, display=False):
    all_champs = dict()

    for base in imagelist: 
        base = cv2.imread(base)
        grey = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)

        smoothed = base // 4
        smoothed = smoothed * 4
        rgbthresh = cv2.inRange(smoothed, np.array([30, 22, 18]),
            np.array([80, 72, 48]))
        rgbaug = cv2.filter2D(rgbthresh, -1, np.ones((2, 2))) 

        contours, h = cv2.findContours(rgbaug, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        boxes = dict()
        polies = list()

        for cnt in contours:
            arclength = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,arclength,True)

            if(len(approx) == 4):
                (x, y, w, h) = cv2.boundingRect(approx)

                ar = w * h
                if ar > 1000:
                    if ar not in boxes:
                        boxes[ar] = list()

                    polies.append(ar)
                    boxes[ar].append([y, x, w, h])

        freqs = Counter(polies)
        freqs = freqs.most_common()
        base_ar = freqs[0][0]

        # merge similar areas
        freqs = {x for x in polies if abs(base_ar - x) < base_ar / 20}

        imboxes = list()
        for ar in freqs:
            for idx, bbox in enumerate(boxes[ar]):
                imboxes.append(base[bbox[0]:bbox[0] + bbox[2],
                    bbox[1]:bbox[1] + bbox[3]])

        corrs = list_corr(imboxes, "champs/")

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

    
def getData():
    res = geturl("https://leagueoflegends.fandom.com/wiki/List_of_champions")    
    soup = bs(res.content, 'lxml')
    table = soup.find_all("table", {"class" : "sortable"})[0]
    df = pd.read_html(str(table))[0]
    
    for idx, row in df.iterrows():
        name = row["Champion"].replace(",", "\xa0").split("\xa0")[0]

        df.at[idx, "Champion"] = name

    return df

def convChamp(champ, cdata):
    EXCHANGE = {
        450 : 90,
        1350 : 270,
        3150 : 630,
        4800 : 960,
        6300 : 1260,
        7800 : 1560
        }

    champ = os.path.splitext(os.path.basename(champ))[0]
    row = cdata[cdata["Champion"].str.lower() == champ]

    cost = int(EXCHANGE[row["Blue Essence"].values[0]])
    name = row["Champion"].values[0]
    return cost, name

def getNumberBox(img):
    boundsy = np.int32(N_BOUNDS[:2] * img.shape[1])
    boundsx = np.int32(N_BOUNDS[2:] * img.shape[0])
    return img[boundsy[0]:boundsy[1], boundsx[0]:boundsx[1]]

def getDigit(image):
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

