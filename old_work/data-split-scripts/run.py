# Test script to list contents of the directory
import os

dir_path = '../DAiSEE/Features/Test/500044/'
#/DAiSEE/Features/Test/5000441001
#r'C:/Users/ahmad/Desktop/EngagementML/DAiSEE/Features/Test/5000441001'
try:
    files = os.listdir(dir_path)
    print("Files in directory:", files)
except Exception as e:
    print("Error accessing directory:", e)
