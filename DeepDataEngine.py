import numpy as np
import cv2
import pickle
import csv
import os
import matplotlib.image as mpimg

class DeepDataEngine:
    """
    Class contains main functions for data management.
    """

    def __init__(
        self,
        set_name, # Name of data set, like 'train' or 'valid'
        storage_dir = './deep_storage', # Folder where data files will be stored
        mem_size = 512 * 1024 * 1024, # Desired maximum size of each file in data storage
        batch_size = 256, # Batch size (used in training and validation process)
        storage_measure_filter = [0, 3]): # Indicates what measures (steering angle, speed) are used for model training
        """
        Initialize class instance
        """

        self.set_name = set_name
        self.storage_dir = storage_dir
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.storage_files = []
        self.storage_file_active = -1
        self.storage_buf_x = None
        self.storage_buf_y = None
        self.storage_measure_filter = storage_measure_filter
        self.storage_size = -1

    def createGenerationPlanFromCSV(
        file_path, # driving_log.csv file path
        angle_shift_clr = (0.0, 0.18, 0.18), # For center, left, right cameras - steering angle shift
        speed_change_clr = (0.0, 0.35, 0.35), # For center, left, right cameras - speed adjustment
        dist_shift_power = 1.25, # Define form of curve for angle/speed shift from side cameras to center.
        strades_ias = (1, 2, 3), # Define strade for images, angle and speed. For images mean skip several records in driving_log.csv file, for angle and speed - shift between image and measure
        augment_prob = 0.3): # Augmentation probability - with this probability center camera data is augmented with side cameras data.
        """
        Static method, creates generation plan from driving_log.csv file.
        """

        generation_plan = []
        generation_plan_buf = []

        image_strade_step = strades_ias[0] - 1

        with open(file_path) as csvfile:
            reader = csv.reader(csvfile)

            for line in reader:
                image_strade_step += 1

                if image_strade_step >= strades_ias[0]:
                    include_in_plan = True
                    image_strade_step = 0
                else:
                    include_in_plan = False

                img_center = line[0]
                img_left = line[1]
                img_right = line[2]
                measure_angle = float(line[3])
                measure_throttle = float(line[4])
                measure_break = float(line[5])
                measure_speed = float(line[6])

                for generation_plan_buf_line in generation_plan_buf:
                    for idx in range(len(generation_plan_buf_line[-1])):
                        generation_plan_buf_line[-1][idx] -= 1
                        if generation_plan_buf_line[-1][idx] == 0:
                            if idx == 0:
                                generation_plan_buf_line[4][0] = measure_angle
                            elif idx == 1:
                                generation_plan_buf_line[4][3] = measure_speed

                if include_in_plan:
                    file_time_stamp = img_center.split('/')[-1][-27:-4].split('_')

                    generation_plan_buf += [[file_time_stamp, ('C', 'N'), (0.0, angle_shift_clr[0], speed_change_clr[0], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]
                    generation_plan_buf += [[file_time_stamp, ('C', 'F'), (0.0, angle_shift_clr[0], speed_change_clr[0], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]

                    if np.random.random() < augment_prob:
                        if np.random.random() < 0.5:
                            generation_plan_buf += [[file_time_stamp, ('L', 'N'), (np.random.random(), angle_shift_clr[1], speed_change_clr[1], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]
                            generation_plan_buf += [[file_time_stamp, ('L', 'F'), (np.random.random(), angle_shift_clr[1], speed_change_clr[1], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]
                        else:
                            generation_plan_buf += [[file_time_stamp, ('R', 'N'), (np.random.random(), angle_shift_clr[2], speed_change_clr[2], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]
                            generation_plan_buf += [[file_time_stamp, ('R', 'F'), (np.random.random(), angle_shift_clr[2], speed_change_clr[2], dist_shift_power), (img_center, img_left, img_right), [measure_angle, measure_throttle, measure_break, measure_speed], [strades_ias[1] - 1, strades_ias[2] - 1]]]

                generation_plan_buf_new = []
                for generation_plan_buf_line in generation_plan_buf:
                    ready_to_plan = True
                    for idx in range(len(generation_plan_buf_line[-1])):
                        if generation_plan_buf_line[-1][idx] > 0:
                            ready_to_plan = False
                            break;

                    if ready_to_plan:
                        generation_plan += [generation_plan_buf_line[:-1]]
                    else:
                        generation_plan_buf_new += [generation_plan_buf_line]

                generation_plan_buf = generation_plan_buf_new

        return generation_plan

    def _unpickleFromFile(self, file_path):
        """
        Unpickle file with data.
        """

        with open(file_path, mode='rb') as f:
            data_set = pickle.load(f)
    
        X_data, y_data = data_set['features'], data_set['labels']

        assert(len(X_data) == len(y_data))

        return X_data, y_data

    def _pickleToFile(self, file_path, X_data, y_data):
        """
        Pickle file with data.
        """

        with open(file_path, mode='wb') as f:
            data_set = {'features' : X_data, 'labels' : y_data}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _unpickleStorageSize(self):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        storage_size = 0

        try:
            with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='rb') as f:
                data_set = pickle.load(f)
    
            storage_size = data_set['storage_size']
        except:
            pass

        return storage_size

    def _pickleStorageSize(self, storage_size):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='wb') as f:
            data_set = {'storage_size' : storage_size}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _loadStorage(self):
        """
        Load information about data storage - size and files with data.
        In this way storage is initialized for reading.
        """

        self.storage_files = []
        self.storage_file_active = -1

        set_file_base_name = self.set_name + '_';

        try:
            os.makedirs(self.storage_dir)
        except:
            pass

        try:
            for file_name in os.listdir(self.storage_dir):
                file_path = self.storage_dir + '/' + file_name
                if (os.path.exists(file_path) and
                    os.path.isfile(file_path) and
                    (str(os.path.splitext(file_path)[1]).upper() in ('.DAT')) and
                    (str(file_name[:len(set_file_base_name)]).upper() == str(set_file_base_name).upper())):
                    
                    self.storage_files += [file_path]

        except:
            pass

        self.storage_size = self._unpickleStorageSize()

    def _delete_storage(self):
        """
        Delete data storage.
        """

        for file_name in self.storage_files:
            try:
                os.remove(file_name)
            except:
                pass

        self.storage_files = []
        self.storage_size = 0
        self._pickleStorageSize(self.storage_size)

    def initStorage(self):
        """
        Initialize storage for reading, call _loadStorage().
        """

        self._loadStorage()

    def createStorage(
        self,
        generation_plan, # Generation plan - python list
        override = True): # Indicates that old storage must be deleted if exists. Otherwise it will be augmented with new files.
        """
        Create data storage from generation plan.
        """

        if len(generation_plan) <= 0:
            return

        self._loadStorage()

        if override:
            self._delete_storage()

        # In case storage already have some data, find index of next file.
        file_idx = -1
        for file_name in self.storage_files:
            cur_idx = int(file_name[-10:-4])
            file_idx = max(file_idx, cur_idx)

        file_idx += 1

        # Read first image to determine shape
        image = cv2.imread(generation_plan[0][3][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape
        measure_shape = len(generation_plan[0][4])
        
        # Create empty X and Y buffers of fixed size. These buffers will be populated with inbound and outbound data and pickled to disk.
        buf_size = int(self.mem_size / ((image_shape[0] * image_shape[1] * image_shape[2]) * 1))

        x_buf = np.zeros((buf_size, image_shape[0], image_shape[1], image_shape[2]), dtype = np.uint8)
        y_buf = np.zeros((buf_size, measure_shape), dtype = np.float32)

        shift_px_const = 29 # shift 29 pixels
        
        # Shuffle generation plan to have random distribution across all data files.
        np.random.shuffle(generation_plan)
        
        buf_pos = 0

        for plan_line in generation_plan:
            # Load metadata and measures from generation plan
            img_action, img_flip = plan_line[1]
            
            img_shift, angle_shift, speed_change, dist_shift_power = plan_line[2]
            img_shift_px = int(img_shift * shift_px_const)
            if img_shift_px == 0:
                img_shift_px = shift_px_const
                img_shift = 1.0
            
            # Load images from disk
            image_center = cv2.imread(plan_line[3][0])
            image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
            
            if img_action == 'L':
                image_shift = cv2.imread(plan_line[3][1])
                image_shift = cv2.cvtColor(image_shift, cv2.COLOR_BGR2RGB)

            if img_action == 'R':
                image_shift = cv2.imread(plan_line[3][2])
                image_shift = cv2.cvtColor(image_shift, cv2.COLOR_BGR2RGB)

            measures = plan_line[4]

            if img_action == 'C':
                res_image = image_center

            if (img_action == 'L') or (img_action == 'R'):
                # For each augmented image calculate angle/speed shift based on point between side camera and center camera, extreme values and shape of curve (power function).
                angle_shift_adj = angle_shift * ((img_shift_px / shift_px_const)**dist_shift_power)
                speed_change_adj = speed_change * ((img_shift_px / shift_px_const)**dist_shift_power)

            if img_action == 'L':
                # Combine image between left and center camera.
                res_image = image_shift
                measures[0] += angle_shift_adj
                measures[3] = (1.0 - speed_change_adj) * measures[3]
                if img_shift_px < shift_px_const:
                    res_image[:, :(img_shift_px - shift_px_const), :] = res_image[:, (shift_px_const - img_shift_px):, :]
                    res_image[:, (img_shift_px - shift_px_const):, :] = image_center[:, -shift_px_const:-img_shift_px, :]

            if img_action == 'R':
                # Combine image between right and center camera.
                res_image = image_shift
                measures[3] = (1.0 - speed_change_adj) * measures[3]
                if img_shift_px < shift_px_const:
                    res_image[:, (shift_px_const - img_shift_px):, :] = res_image[:, :(img_shift_px - shift_px_const), :]
                    res_image[:, :(shift_px_const - img_shift_px), :] = image_center[:, img_shift_px:shift_px_const, :]

            if img_flip == 'F':
                # Flip image
                res_image = cv2.flip(res_image, 1)
                measures[0] = -measures[0]

            measures[3] = (measures[3] - 15.0) / 100.0 #Scale speed

            x_buf[buf_pos] = res_image
            y_buf[buf_pos] = measures
                        
            buf_pos += 1

            if buf_pos >= buf_size:
                # Pickle buffer to file
                self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
                self.storage_size += buf_size
                self._pickleStorageSize(self.storage_size)
                file_idx += 1
                buf_pos = 0

        if buf_pos > 0:
            # Pickle non-full last buffer to file
            x_buf = x_buf[:buf_pos]
            y_buf = y_buf[:buf_pos]
            self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
            self.storage_size += buf_pos
            self._pickleStorageSize(self.storage_size)

        # Initialize storage for reading
        self._loadStorage()

    def _readNextStorageFile(self):
        """
        Read next storage file from disk.
        """

        self.storage_buf_x, self.storage_buf_y = self._unpickleFromFile(self.storage_files[self.storage_file_active])

        if self.storage_measure_filter != None:
            self.storage_buf_y = self.storage_buf_y[:, self.storage_measure_filter]

        permutation = np.random.permutation(len(self.storage_buf_x))
        self.storage_buf_x = self.storage_buf_x[permutation]
        self.storage_buf_y = self.storage_buf_y[permutation]

    def initRead(self):
        """
        Initialize data reading - shuffle file list and read next non-empty file.
        """

        np.random.shuffle(self.storage_files)
        self.storage_file_active = 0
        self._readNextStorageFile()

        while len(self.storage_buf_x) <= 0:
            if (self.storage_file_active + 1) < len(self.storage_files):
                self.storage_file_active += 1
                self._readNextStorageFile()
            else:
                break

    def canReadMore(self):
        """
        Determine that data storage is fully read and to read more need be initialized with initRead() function.
        """

        return len(self.storage_buf_x) > 0

    def readNext(self):
        """
        Read next batch for training or validation.
        If end of current file is reached, next file is automatically read from disk and append to current buffer.
        Only one last buffer per epoch can have size less that batch_size.
        """

        x_data = np.array(self.storage_buf_x[:self.batch_size])
        y_data = np.array(self.storage_buf_y[:self.batch_size])

        batch_buf_size = len(x_data)
        self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
        self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

        try_read_next = True

        while try_read_next:
            try_read_next = False

            if len(self.storage_buf_x) <= 0:
                if (self.storage_file_active + 1) < len(self.storage_files):
                    self.storage_file_active += 1
                    self._readNextStorageFile()

                    if len(self.storage_buf_x) > 0:
                        if len(x_data) <= 0:
                            x_data = np.array(self.storage_buf_x[:self.batch_size])
                            y_data = np.array(self.storage_buf_y[:self.batch_size])

                            batch_buf_size = len(x_data)
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]
                        elif len(x_data) < self.batch_size:
                            size_orig = len(x_data)
                            batch_remain = self.batch_size - size_orig
                            x_data = np.append(x_data, np.array(self.storage_buf_x[:batch_remain]), axis = 0)
                            y_data = np.append(y_data, np.array(self.storage_buf_y[:batch_remain]), axis = 0)

                            batch_buf_size = len(x_data) - size_orig
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

                    if len(self.storage_buf_x) <= 0:
                        try_read_next = True

        return x_data, y_data

    def _generator(self):
        """
        Infinite generator compatible with Keras
        """

        while True:
            self.initRead()
            while self.canReadMore():
                yield self.readNext()

    def getGenerator(self):
        """
        Return number of unique batches can be read per epoch and generator instance. Compatible with Keras.
        """

        gen_step_max = self.storage_size // self.batch_size
        if (self.storage_size % self.batch_size) > 0:
            gen_step_max += 1

        return gen_step_max, self._generator()

    def getInOutShape(self):
        """
        Get shape of input and output data.
        """

        self.initRead()
        if self.canReadMore():
            x_data, y_data = self.readNext()
            return x_data.shape[1:], y_data.shape[1:]

        return (), ()