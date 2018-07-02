#!pip install whoosh
#!pip install gensim


import os
import urllib
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import csv
import json
import numpy as np
from tqdm import tqdm_notebook
import sys
import re
import ast
import collections
import time
import shutil
import ast
import gzip
import heapq
from operator import itemgetter
import collections
from whoosh.analysis import CharsetFilter, StemmingAnalyzer
from whoosh import fields
from whoosh.support.charset import accent_map
from string import ascii_letters
import pickle
import math


# Artist titles that are not to be considered as artists
nop_titles = ["alt","mo",## NEW ADITIONS BY SANTI
              "car", "dream", "arizona", "game", "america", "fun",  "pain", "jack", 'joe', 'lit', 'big',
              'flame','monica','heavi','fuego','sweet', 'bath','shine','dope','reggaeton','girl','babi','good morn'
            , 'club','mis','week','todai','idk','love','hous','why','mood','jam','top','alex','almost',
             'low','hurt','water','junior','alex', 'sundai','beast','me','church','sound','peac','time',
             'amber','control', 'dad','ic','boogi','thursdai','dig','holidai','eclips','ye','weeknd',
             'cold','main','lover','emot','boi','camp','bone','beat','magic','home','live','savag','cream',
             'ha','want','him','do','green','logic','lord','citi','moment','mountain','karen','peach', 'next','root'
             ]


def normalize_name(name):
    stem = True
    letters = list(name)
    
    # if format w o r k o u t / w.o.r.k.o.u.t/ w*o*r*k*o*u*t join togother
    if len(letters)>4:
        if len(set([letters[i] for i in range(0,len(letters),2)]))==1:
            name = "".join([letters[i] for i in range(1,len(letters),2)])
        elif len(set([letters[i] for i in range(1,len(letters),2)]))==1:
            name = "".join([letters[i] for i in range(0,len(letters),2)])
             
    # if there is and & not surrounded by spaces, leave alone (example 'r&b)
    if "&" in letters:
        position = letters.index("&")
        if position>0 and position<len(letters)-1:
            if letters[position-1]!=' ' and letters[position+1]!=' ':
                stem  = False
    
      
    # if there is a k surrounded by numbers turn to 0
    if "k" in letters and '2' in letters:
        positions = [x for x in range(len(letters)) if letters[x]=='k']
        for pos in positions:
             if pos>0 and pos<len(letters)-1:
                if letters[pos-1]=='2':
                    letters[pos]='0'
                    name = "".join(letters)
           
    # proceed to stem   
    if stem: 
        my_analyzer = StemmingAnalyzer() | CharsetFilter(accent_map)
        tokens = my_analyzer(name)
        words = [token.text for token in tokens]
        
        # if the reuslt is empyt, leave alone, if not, return as a list
        if len(words)!=0:
            result=""
            for el in words:
                result +=el+" "
            letters = list(result)[:-1]
    # softer stem
    else:
        name = name.lower()
        name = re.sub(r"[.,'\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        letters = list(name)
        
            
            
    # if last n characters are equal leave only 1 
    last = letters[-1]
    if last in ascii_letters and len(letters)>1:
        while(letters[-2]==last):
            letters.pop(-2)
            if len(letters)==1: break
    
    
    return ''.join(letters)

            
def evaluate(solution,recommendations):
    result = []
    for pid in recommendations:
        result.append( len([t for t in recommendations[pid] if t in solution[str(pid)]]) / len(solution[str(pid)]) )        
    print(round(sum(result)/len(result),3))
    return round(sum(result)/len(result),3)


# load challenge set            
def load_challenge_set():
    path_challenge_set = "../challenge_set.json"
    f = open(path_challenge_set)
    js = f.read()
    f.close()
    challange_set = json.loads(js)
    return challange_set

    
    
# recommends tracks to original challange set and writes them in a csv
def process_challenge_set(recsys_method,name,dict_trackuri_id,dict_id_trackuri,title_tracks,params):
    
    challange_set = load_challenge_set()
    recommendations = {}  # recommendations in ID given by model
    seeds = {}
    
    for playlist in tqdm_notebook(challange_set["playlists"]):
        recommendations = recsys_method(playlist, title_tracks,recommendations,params)
        seeds[playlist["pid"]]=[ str(dict_trackuri_id[track["track_uri"]]) for track in playlist["tracks"]] 
             
    final_recomendations = result_to_trackuri(recommendations,seeds,dict_id_trackuri)
    write_csv(name,final_recomendations) 

    

# convert id to trackruri and delete seeds and repeated tracks
# this function orders the tracks by popularity (id) before converting them to track uri 
def result_to_trackuri(recommendations,seeds,dict_id_trackuri):
    final = {}
    final_ids = {}
    for pid in recommendations:
        if pid in seeds: 
            final[pid]=[]
            final_ids[pid] = []
            n_tracks = 0
            for track in recommendations[pid]:
                    if str(track) not in seeds[pid]:              #check the track not in the original playlist  
                        if str(track) in dict_id_trackuri:        #check that the track is in the ID track uri
                            if dict_id_trackuri[str(track)] not in final[pid]:   #check that it is not repeated
                                final[pid].append(dict_id_trackuri[str(track)])
                                final_ids[pid].append(track)
                                n_tracks +=1
                                if(n_tracks>501):
                                    break
            final_ids[pid] = final_ids[pid][:500]
            final[pid] = final[pid][:500]                         #only 500 tracks needed
        
        final_ids[pid] = [int(i) for i in final_ids[pid]] #to int and 
        
        final[pid] = [dict_id_trackuri[str(track)] for track in final_ids[pid]]  #to string and to track uri 
    return final
    