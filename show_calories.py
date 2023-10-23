# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:04:30 2019

@author: anas
"""

import xml.etree.ElementTree as ET
tree = ET.parse('calories_of_food.xml')
root = tree.getroot()

class_num=1    
print (root[class_num].attrib['name'])
print (root[class_num][0].tag,'=',root[class_num][0].text,' kcal')
print (root[class_num][1].tag,'=',root[class_num][1].text,' kcal')
print (root[class_num][2].tag,'=',root[class_num][2].text,' kcal')
