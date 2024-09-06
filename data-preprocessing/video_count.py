import os

'''
This script counts the number of video files in a given dataset directory. The purpose of this script is to help calculate 
the total number of frames extracted from these videos, which in turn allows us to determine the necessary amount of features 
required for training a machine learning model. 

By knowing the total number of videos, we can estimate the total number of frames and the corresponding storage requirements 
for preprocessed data and feature extraction steps. 
'''

def count_videos(directory):
    video_extensions = ('.avi', '.mp4', '.mov')  
    video_count = 0

    for files in os.walk(directory):
        for file in files:
            if file.endswith(video_extensions):
                video_count += 1

    return video_count

# Replace the path below with the path to your directory
directory_path = 'C://Users//ahmad//Desktop//EngagementML//DAiSEE//DataSet'
total_videos = count_videos(directory_path)
print(f'Total video files: {total_videos}')
