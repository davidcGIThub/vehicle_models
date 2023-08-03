import numpy as np

class BsplineEvaluator:

    def __init__(self, order, initial_num_points_to_check_per_interval = 100, 
                 num_points_to_check_per_interval = 10000):
        self._order = order
        self._initial_num_points_to_check_per_interval = initial_num_points_to_check_per_interval
        self._num_points_to_check_per_interval = num_points_to_check_per_interval

    def get_closest_point_and_derivatives(self, control_points, scale_factor, position):
        control_point_set = self.get_closest_control_point_set(control_points, position)
        closest_point, t = self.get_closest_point_and_t(control_point_set, scale_factor, position)
        velocity_vector = self.__get_velocity_vector(t, control_point_set, scale_factor)
        acceleration_vector = self.__get_acceleration_vector(t, control_point_set, scale_factor)
        return closest_point, velocity_vector, acceleration_vector

    def get_closest_control_point_set(self, control_points, position):
        num_control_points = self.__count_number_of_control_points(control_points)
        dataset = self.matrix_bspline_evaluation_for_dataset(control_points, 
                        self._initial_num_points_to_check_per_interval)
        distances = np.linalg.norm(position.flatten()[:,None] - dataset,2,0)
        num_intervals = num_control_points - self._order
        intial_ctrl_pt_index = int(np.argmin(distances)/len(distances)*num_intervals)
        control_point_set = control_points[:,intial_ctrl_pt_index:intial_ctrl_pt_index+self._order+1]
        return control_point_set

    def get_closest_point_and_t(self, control_points, scale_factor, position):
        dataset = self.matrix_bspline_evaluation_for_dataset(control_points, 
                        self._num_points_to_check_per_interval)
        distances = np.linalg.norm(position.flatten()[:,None] - dataset,2,0)
        closest_point_index = np.argmin(distances)
        closest_point = dataset[:,closest_point_index][:,None]
        t = closest_point_index/len(distances)
        return closest_point, t*scale_factor
    
    def matrix_bspline_evaluation_for_dataset(self, control_points, num_points_per_interval):
        """
        This function evaluates the B spline for a given time data-set
        """
        #initialize variables
        num_ppi = num_points_per_interval
        dimension = self.__get_dimension(control_points)
        number_of_control_points = self.__count_number_of_control_points(control_points)
        num_intervals = number_of_control_points - self._order
        #create steps matrix
        steps_array = np.linspace(0,1,num_ppi+1)
        L = np.ones((self._order+1,num_ppi+1))
        for i in range(self._order+1):
            L[i,:] = steps_array**(self._order-i)
        # Find M matrix
        M = self.__get_M_matrix(self._order)
        #Evaluate spline data
        if dimension > 1:
            spline_data = np.zeros((dimension,num_intervals*num_ppi+1))
        else:
            spline_data = np.zeros(num_intervals*num_ppi+1)
        for i in range(num_intervals):
            if dimension > 1:
                P = control_points[:,i:i+self._order+1]
            else:
                P = control_points[i:i+self._order+1]
            spline_data_over_interval = np.dot(np.dot(P,M),L)
            if dimension > 1:
                if i == num_intervals-1:
                    spline_data[:,i*num_ppi:(i+1)*num_ppi+1] = spline_data_over_interval[:,0:num_ppi+1]
                else:
                    spline_data[:,i*num_ppi:(i+1)*num_ppi] = spline_data_over_interval[:,0:num_ppi]
            else:
                if i == num_intervals-1:
                    spline_data[i*num_ppi:(i+1)*num_ppi+1] = spline_data_over_interval[0:num_ppi+1]
                else:
                    spline_data[i*num_ppi:(i+1)*num_ppi] = spline_data_over_interval[0:num_ppi]
        return spline_data
    
    def get_distance_to_endpoint(self, control_points: np.ndarray, position: np.ndarray):
        end_control_points = control_points[:,-(self._order+1):]
        M = self.__get_M_matrix(self._order)
        T = self.__get_T_vector(self._order, 1,0,1)
        end_point = np.dot(end_control_points,np.dot(M,T))
        distance = np.linalg.norm(end_point.flatten() - position.flatten())
        return distance
    
    def get_distance_to_startpoint(self, control_points: np.ndarray, position: np.ndarray):
        start_control_points = control_points[:,0:self._order+1]
        M = self.__get_M_matrix(self._order)
        T = self.__get_T_vector(self._order, 0,0,1)
        start_point = np.dot(start_control_points,np.dot(M,T))
        distance = np.linalg.norm(start_point.flatten() - position.flatten())
        return distance
    
    def __get_velocity_vector(self, t, control_point_set, scale_factor):
        M = self.__get_M_matrix(self._order)
        T = self.__get_T_derivative_vector(self._order,t,0,1,scale_factor)
        velocity_vector = np.dot(control_point_set, np.dot(M,T))
        return velocity_vector
    
    def __get_acceleration_vector(self, t, control_point_set, scale_factor):
        M = self.__get_M_matrix(self._order)
        T = self.__get_T_derivative_vector(self._order,t,0,2,scale_factor)
        acceleration_vector = np.dot(control_point_set, np.dot(M,T))
        return acceleration_vector

    def __count_number_of_control_points(self, control_points):
        if control_points.ndim == 1:
            number_of_control_points = len(control_points)
        else:
            number_of_control_points = len(control_points[0])
        return number_of_control_points
    
    def __get_dimension(self, control_points):
        if control_points.ndim == 1:
            dimension = 1
        else:
            dimension = len(control_points)
        return dimension

    def __get_M_matrix(self, order):
        if order > 5:
            print("Error: Cannot compute higher than 5th order matrix evaluation")
            return None
        if order == 0:
            return 1
        if order == 1:
            M = self.__get_1_order_matrix()
        if order == 2:
            M = self.__get_2_order_matrix()
        elif order == 3:
            M = self.__get_3_order_matrix()
        elif order == 4:
            M = self.__get_4_order_matrix()
        elif order == 5:
            M = self.__get_5_order_matrix()
        return M

    def __get_T_derivative_vector(self, order,t,tj,rth_derivative,scale_factor):
        T = np.zeros((order+1,1))
        t_tj = t-tj
        for i in range(order-rth_derivative+1):
            T[i,0] = (t_tj**(order-rth_derivative-i))/(scale_factor**(order-i)) * np.math.factorial(order-i)/np.math.factorial(order-i-rth_derivative)
        return T

    def __get_T_vector(self, order,t,tj,scale_factor):
        T = np.ones((order+1,1))
        t_tj = t-tj
        for i in range(order+1):
            if i > order:
                T[i,0] = 0
            else:
                T[i,0] = (t_tj/scale_factor)**(order-i)
        return T

    def __get_1_order_matrix(self):
        M = np.array([[-1,1],
                        [1,0]])
        return M

    def __get_2_order_matrix(self):
        M = .5*np.array([[1,-2,1],
                            [-2,2,1],
                            [1,0,0]])
        return M

    def __get_3_order_matrix(self):
        M = np.array([[-2 ,  6 , -6 , 2],
                        [ 6 , -12 ,  0 , 8],
                        [-6 ,  6 ,  6 , 2],
                        [ 2 ,  0 ,  0 , 0]])/12
        return M

    def __get_4_order_matrix(self):
        M = np.array([[ 1 , -4  ,  6 , -4  , 1],
                        [-4 ,  12 , -6 , -12 , 11],
                        [ 6 , -12 , -6 ,  12 , 11],
                        [-4 ,  4  ,  6 ,  4  , 1],
                        [ 1 ,  0  ,  0 ,  0  , 0]])/24
        return M

    def __get_5_order_matrix(self):
        M = np.array([[-1  ,  5  , -10 ,  10 , -5  , 1],
                        [ 5  , -20 ,  20 ,  20 , -50 , 26],
                        [-10 ,  30 ,  0  , -60 ,  0  , 66],
                        [ 10 , -20 , -20 ,  20 ,  50 , 26],
                        [-5  ,  5  ,  10 ,  10 ,  5  , 1 ],
                        [ 1  ,  0  ,  0  ,  0  ,  0  , 0]])/120
        return M

