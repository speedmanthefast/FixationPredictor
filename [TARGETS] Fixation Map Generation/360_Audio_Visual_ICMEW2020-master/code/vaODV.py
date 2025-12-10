import imageio
import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path

FIXATION_FOLDER = 'output/fixations/'

class Point:
    def __init__(self, posX, posY, registeredTimes):
        self.ClId = 0
        self.times = np.array(registeredTimes)
        self.pos = posX * posY
        self.posX= posX
        self.posY= posY

    def ChangeClId(self, ClId):
        self.ClId = ClId

class vaODV:
    def __init__(self, odv_folder, user_dataset, modality, odv_shape=None):
        # modaliy
        self.modality = modality

        # odv folder
        self.odv_folder = odv_folder
        # odv path
        self.userdata_path = self.get_userdata_path(user_dataset, modality)
        # list odvs
        self.odv_list = self.get_ODVs()

        # given shape
        self.odv_shape = odv_shape

        # odv metadata
        self.vid_info = {}


    def get_ODVs(self):
        list_vid = [k for k in glob.glob(os.path.join(self.userdata_path, '*/'))]
        return list_vid

    def get_userdata_path(self, user_dataset, modality):
        st_fixation_folder = os.path.join(str(user_dataset)+'/', modality)
        return st_fixation_folder


    def get_odvInfo(self, odv_name):

        vid = imageio.get_reader(os.path.join(self.odv_folder, odv_name + '.mp4'), 'ffmpeg')

        self.vid_info['nframes']    = vid.get_meta_data()['nframes']
        self.vid_info['size']       = vid.get_meta_data()['size']
        self.vid_info['duration']   = int(vid.get_meta_data()['duration'])
        self.vid_info['fps']        = vid.get_meta_data()['fps']

        if self.odv_shape is None:
            self.odv_shape =  [self.vid_info['size'][1], self.vid_info['size'][0], 3]


    def get_participantNumber(self, odv_name):
        return glob.glob(os.path.join(self.userdata_path, odv_name+'/*.csv'))

    def display_status(self, count, odv_name):
        print("Modality: {modality}    ODV: {odv_name}   {f_count}/{list_vid}"
        .format(modality=self.modality, odv_name=odv_name, f_count=count+1, list_vid=len(self.odv_list)))

    def filter_par(self, par, f, f_next=1):
        df      = pd.read_csv(par)
        # Filters data between current start time (f) and next increment (f+f_next)
        _filter = (df['time']<(f+f_next)) & (df['time']>=f)

        return df[_filter]

    def RegionQuery(self, setOfPoints, point, eps):
        seeds = []

        # We append the point as many times as it appears, it should count for the fixations
        pointTimes = point.times # This is an np.array :)
        for k in range(1, len(pointTimes)):
            seeds.append(Point(point.posX, point.posY, pointTimes[k]))

        for i in range(eps):
            for j in range(eps):

                # All possible combinations in an eps distance from the point
                x1=point.posX + i
                x2=point.posX - i
                y1=point.posY + j
                y2=point.posY - j

                # We append the point as many times as it appears, it should count for the fixations
                if i != 0 or j != 0 :
                    times_x1_y1 = np.array(setOfPoints[x1, y1])
                    if times_x1_y1.size > 1:
                        for k in range( 1, len(times_x1_y1)):
                            seeds.append(Point(x1, y1, times_x1_y1[k]))

                    times_x1_y2 = np.array(setOfPoints[x1, y2])
                    if times_x1_y2.size > 1:
                        for k in range( 1, len(times_x1_y2)):
                            seeds.append(Point(x1, y2, times_x1_y2[k]))

                    times_x2_y1 = np.array(setOfPoints[x2, y1])
                    if times_x2_y1.size > 1:
                        for k in range( 1, len(times_x2_y1)):
                            seeds.append(Point(x2, y1, times_x2_y1[k]))

                    times_x2_y2 = np.array(setOfPoints[x2, y2])
                    if times_x2_y2.size > 1:
                        for k in range( 1, len(times_x2_y2)):
                            seeds.append(Point(x2, y2, times_x2_y2[k]))

        return seeds

    def ExpandCluster(self, setOfPoints, Fixations_person, point, eps, ClId, minPts): # DBSCAN method
        seeds = self.RegionQuery(setOfPoints, point, eps) # returns the eps-neighborhood of point

        if len(seeds) < minPts :
            point.ChangeClId = 1 # Meaning Noise
            return False
        else:
            Fixations_person[point.posX, point.posY] = 1 # Getting only the center point of the fixation
            for i in range(len(seeds)):
                Fixations_person[seeds[i].posX, seeds[i].posY] = 1 # not only the center
                seeds[i].ChangeClId(ClId)

            return True

    def clustering(self, data_par):


        RegisteredPoints_person = np.zeros((self.odv_shape[0], self.odv_shape[1]), dtype=object) # Created to have a heat map per person
        # NOTE: changed dtype=np.int to dtype=int
        ProbMatrix_person       = np.zeros((self.odv_shape[0], self.odv_shape[1]), dtype=int) # Created to have a heat map per person

        # don't consider the first fixation (1s)
        # dataPoints = dataPoints[40:] #ana=40

        array_x    = round(data_par['2dmu']*self.odv_shape[1]).astype(int)
        array_y    = round(data_par['2dmv']*self.odv_shape[0]).astype(int)

        #*********************************************************************************
        # Assigns +1 in a position (x,y) detected
        #*********************************************************************************
        for i in range(len(data_par)):
            x = int(array_x.iloc[i])
            y = int(array_y.iloc[i])
            # print(x,y)
            if (x<self.odv_shape[1] and y<self.odv_shape[0] and x>=0 and y>=0):
                ProbMatrix_person[y,x] += 1
                RegisteredPoints_person[y, x] = np.append(RegisteredPoints_person[y, x], data_par['time'].iloc[i])

        ClusterId = 2 # First ClusterId
        Fixations_person = np.zeros((self.odv_shape[0], self.odv_shape[1])) # Created to have a heat map per person

        for i in range(self.odv_shape[0]):
            for j in range(self.odv_shape[1]):
                point = Point(i, j, RegisteredPoints_person[i,j])

                if point.times.size > 1: # Meaning, it was a visited point

                    xnew = i - int(round(self.odv_shape[0]/2)) # i-height/2 I need the point centered in 0
                    eps     = np.rint(6 * (1 / (np.cos(xnew/int(round(self.odv_shape[0]/2)) * np.pi/2)))) # rounded to the closest integer threshold

                    if (point.posX > eps and  point.posX < (self.odv_shape[0] - eps) and point.posY > eps and point.posY < (self.odv_shape[1] - eps)):

                        if self.ExpandCluster(RegisteredPoints_person, Fixations_person, point, int(eps), ClusterId, minPts=12):
                            ClusterId += 1

        return Fixations_person

    def init_map(self):
        self.fixation_map       = np.zeros((self.odv_shape[0],self.odv_shape[1]))


    def generate_fixations(self, odv_name):
        # get the metadata (vid_info) for a given ODV
        self.get_odvInfo(odv_name)

        # number of participants
        self.participants = self.get_participantNumber(odv_name)

        # create a folder for fixation
        fix_folder = os.path.join(FIXATION_FOLDER + '/' + self.modality, odv_name)
        Path(fix_folder).mkdir(parents=True, exist_ok=True)

        fixation_maps = []

        # CHANGED: Setup for 8 segments per second
        FPS_TARGET = 8
        time_step = 1.0 / FPS_TARGET
        total_segments = int(self.vid_info['duration'] * FPS_TARGET)
        # ------------------------------------------------

        for i in range(total_segments):
            self.init_map()

            # Calculate current start time in seconds
            current_time = i * time_step

            print(":::...Segment {seg} / {total} (Time: {t:.3f}s)".format(
                seg=i, total=total_segments, t=current_time))

            for par in self.participants:
                # print("::. ODV #{}".format(os.path.basename(par))) # Optional: commented out to reduce clutter with 8x more loops

                # --- CHANGED: Pass f_next=time_step to capture only 0.125s of data ---
                data_par = self.filter_par(par, current_time, f_next=time_step)

                # If no data points exist in this tiny slice, skip clustering to save time/errors
                if not data_par.empty:
                    Fixations_person = self.clustering(data_par)
                    self.fixation_map += Fixations_person

            # NOTE: encountered an issue saving the png with float values. Have to normalize the map to 8-bit unsigned int type.
            # Find the maximum value in the map to normalize it
            max_fixation_value = self.fixation_map.max()

            # Create a normalized map, avoiding division by zero if the map is empty
            if max_fixation_value > 0:
                # Scale the map to the 0-255 range
                normalized_map = (255.0 * self.fixation_map / max_fixation_value).astype(np.uint8)
            else:
                # If the map is all zeros, just create an empty uint8 map
                normalized_map = self.fixation_map.astype(np.uint8)

            # Save the properly formatted map
            # CHANGED: naming convention so it can be properly sorted
            imageio.imwrite(os.path.join(fix_folder, f'fixmap_f_{i:05d}.png'), normalized_map)
            fixation_maps.append(self.fixation_map)

        return fixation_maps
