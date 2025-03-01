# -*- coding: utf-8 -*-
"""
PHYS10362 Introduction to Programming - Assignment 2

This code reads in values for the observed wavelength emitted from a star and 
the times at which they were observed from data files, and finds the optimum 
values of v_0 and omega for which the minimum chi-squared is found for
expected wavelength values against the measured values. It also produces 
a plot showing the expected wavelength values and the measured wavelength
values against time, and a contour plot of the reduced chi-squared value for 
different values of v_0 and omega.
(extra note: the contour plot takes a while to run on the university 
computers)
Assignment details can be found on Blackboard > Courses  > 
Introduction to Programming > Assignments > Final Assignment: Doppler 2024
Ashrita Padigala 28/02/2024
"""
import math
from math import floor, log10
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

NUM_DATA_POINTS = 0
THETA = 0

def read_file():
    """
    # reads in files doppler_data_1.csv and doppler_data_2.csv 
    and validates the files 

    :return:
    """
    time_values = np.array([])
    wavelength_values = np.array([])
    uncertainties = np.array([])
    total_list = []
    errors = ["File Errors:"] # list of errors encountered during file reading
    try: # reads in and validates data from first file
        my_file_1 = open('doppler_data_1.csv', 'r')
        for line in my_file_1:
            line_no_return = line.strip('\n')
            line_elements = line_no_return.split(",")
            if line_elements[0] != '%' and len(line_elements) ==3 :
                try:
                    time = float(line_elements[0])
                    wavelength = float(line_elements[1])
                    uncertainty = float(line_elements[2])
                    if (
                            math.isnan(time) is False and
                            math.isnan(wavelength) is False and
                            math.isnan(uncertainty) is False and
                            uncertainty != 0
                        ):
                        # appends the valid data points onto lists
                        time_values = np.append(time_values, time)
                        wavelength_values = np.append\
                            (wavelength_values, wavelength)
                        uncertainties = np.append(uncertainties, uncertainty)
                    # appends the corresponding errors onto a list of errors
                    elif (
                            math.isnan(time) is True or
                            math.isnan(wavelength) is True or
                            math.isnan(uncertainty) is True
                         ):
                        errors.append("Value is NaN") # NaN value error
                    elif uncertainty == 0:
                        errors.append("0 uncertainty for this wavelength:"\
                                      +str(wavelength)) # 0 uncertainty error
                except ValueError as err: # value error (value isn't a float)
                    errors.append("ValueError; "+str(err))
            else: # title line error
                errors.append("This line begins with %"+str(line_no_return))
        my_file_1.close()
    except FileNotFoundError as err:
        errors.append("File not found"+str(err)) # file not found error
    # reads in and validates data from second file (same procedure as 1st file)
    try:
        my_file_2 = open('doppler_data_2.csv', 'r')
        for line in my_file_2:
            line_no_return = line.strip('\n')
            line_elements = line_no_return.split(",")
            if line_elements[0] != '%' and len(line_elements) ==3 :
                try:
                    time = float(line_elements[0])
                    wavelength = float(line_elements[1])
                    uncertainty = float(line_elements[2])
                    if (
                            math.isnan(time) is False and
                            math.isnan(wavelength) is False and
                            math.isnan(uncertainty) is False and
                            uncertainty != 0
                            ):
                        time_values = np.append(time_values, time)
                        wavelength_values = np.append\
                            (wavelength_values, wavelength)
                        uncertainties = np.append(uncertainties, uncertainty)
                    elif (
                            math.isnan(time) is True or
                            math.isnan(wavelength) is True or
                            math.isnan(uncertainty) is True
                            ):
                        errors.append("Value is NaN")
                    elif uncertainty == 0:
                        errors.append\
                            ("0 uncertainty for this wavelength:"+
                             str(wavelength))
                except ValueError as err:
                    errors.append("ValueError; "+str(err))
            else:
                errors.append("This line begins with %"+str(line_no_return))
        my_file_2.close()
    except FileNotFoundError as err:
        errors.append("File not found"+str(err))
    """ converts the data read into SI units and appends each list of
    converted data (and error messages) into a larger list which is returned"""
    time_values = (time_values*31536000).tolist()
    wavelength_values = (wavelength_values*1e-09).tolist()
    uncertainties = (uncertainties*1e-09).tolist()
    total_list.append(time_values)
    total_list.append(wavelength_values)
    total_list.append(uncertainties)
    total_list.append(errors)
    return total_list

def find_minima():
    """
    # finds the values of v_0 and omega that give the minimum
    chi-squared value for the given function 

    :param total_list:
    :return:
    """
    minima = fmin\
        (two_parameter_function, [50,3e-08], full_output = True, disp = False)
    return minima


def find_values(bool_val, total_uncertainties):
    """
    # calculates and prints the different variables involved,
    as well as their uncertainties

    :param bool_val:
    :param total_uncertainties:
    :return:
    """
    total_list = []
    minima = find_minima()
    chi_squared = round(minima[1],3)
    reduced_chi_squared = find_reduced_chi_squared\
        (chi_squared, NUM_DATA_POINTS, 2)
    v_0 = get_v_0(minima[0])
    omega = get_omega(minima[0])
    r_val = sig_figs(get_r(omega), 4)
    m_p = sig_figs(get_m_p(r_val, v_0),4)
    total_list.append(v_0)
    total_list.append(omega)
    total_list.append(reduced_chi_squared)
    total_list.append(r_val)
    total_list.append(m_p)
    if bool_val is True and total_uncertainties[0] is not False:
        """this condition is only true when uncertainties have been able
        to be found from contour plot"""
        v0_uncert = sig_figs(total_uncertainties[0], 4)
        omega_uncert = sig_figs(total_uncertainties[1], 4)
        mp_uncert = round(total_uncertainties[2],3)
        radius_uncert = round(total_uncertainties[3], 3)
        print("Chi-Squared:", chi_squared)
        print("Reduced Chi-Squared:", reduced_chi_squared)
        print("V_0 :", v_0, "±", v0_uncert, "m/s")
        print("Omega:", omega, "±", omega_uncert, "Hz")
        print("R :", r_val,  "±", radius_uncert, "AU")
        print("M_p :", m_p, "±", mp_uncert, "Jovian masses")
    elif bool_val is True and total_uncertainties[0] is False :
        """this condition is true when uncertainties have not been able
        to be found from contour plot"""
        print("Chi-Squared:", chi_squared)
        print("Reduced Chi-Squared:", reduced_chi_squared)
        print("V_0 :", v_0, "m/s")
        print("Omega :", omega, "Hz")
        print("R :", r_val, "AU")
        print("M_p :", m_p,"Jovian masses")
    return total_list

def two_parameter_function(x_y):
    """
    # function for separating out tuple with v_0 and omega values 
    into separate variables to pass into the function to find chi-squared

    :param xy:
    :return:
    """
    x_v = x_y[0]
    y_v = x_y[1]
    return find_chi_squared(x_v, y_v)

def find_chi_squared(x_v, y_v):
    """
    # finds the chi-squared value for the given v_0 and omega values 

    :param xv:
    :param yv:
    :return:
    """
    global NUM_DATA_POINTS
    total_list = read_file()
    outlier_array = np.array([])
    chi_squared = 0
    time_values = total_list[0]
    wavelength_values = total_list[1]
    uncertainties = total_list[2]
    lambda_0 = 656.281e-09
    phi = 0
    NUM_DATA_POINTS = len(time_values)
    for i in range (len(time_values)):
        """iterates through data to find the wavelength 
        expected for each time value"""
        wavelength_measured = wavelength_values[i]
        # finds expected wavelength for the current omega and v_0 values
        wavelength_expected = \
            wavelength_expected_function(x_v, time_values[i], phi, y_v, lambda_0)
        # difference between the measured & corresponding expected wavelength
        outlier = abs(wavelength_measured - wavelength_expected)
        # appends the differences onto an array
        outlier_array = np.append(outlier_array, outlier)
    outlier_array = np.sort(outlier_array) # sorts array of differences
    """sets outlier as the 6th largest difference between expected and
    measured wavelength"""
    max_outlier = outlier_array[-6]
    for i in range(len(time_values)): # iterates through all data
        wavelength_measured = wavelength_values[i]
        # finds the expected wavelength for the current omega and v_0 values
        wavelength_expected = \
            wavelength_expected_function(x_v, time_values[i], phi, y_v, lambda_0)
        """for the current data point, if the difference between expected
        and measured wavelength is bigger than the defined outlier,
        the data point is not considered"""
        outlier = abs(wavelength_measured - wavelength_expected)
        if  outlier < max_outlier:
            # calculates chi-squared taking into account every valid data point
            chi_squared = chi_squared +\
                ((wavelength_measured - wavelength_expected )\
                 /uncertainties[i])**2
        else:
            """for outlier values, since they're not considered,
            total number of data points decreases by 1"""
            NUM_DATA_POINTS = NUM_DATA_POINTS - 1
    return chi_squared

def produce_plot(time_values, wavelength_values, v_0, omega, uncertainties):
    """
    # produces a plot of the data read in (wavelength observed against time) 
    and the calculated wavelength against time for the optimum 
    v_0 and omega values

    :param time_values:
    :param wavelength_values:
    :param v_0:
    :param omega:
    :param uncertainties:
    """
    c_value = 3e08
    lambda_0 = 656.281e-09
    xv_array = np.array(time_values)
    y_values = np.array(wavelength_values)
    uncertainties = np.array(uncertainties)
    y_2_values = ((c_value+(v_0*np.sin\
                            (omega*xv_array+16.83*math.pi/18)))/c_value)\
        *lambda_0
    new_array_y = np.array([])
    new_array_x = np.array([])
    new_array_uncerts = np.array([])
    ordered = np.sort(abs(y_values - y_2_values))
    max_outlier = ordered[-6]
    for i in range(len(xv_array)):
        wavelength_measured = y_values[i]
        wavelength_expected = y_2_values[i]
        outlier = abs(wavelength_measured - wavelength_expected)
        if  outlier < max_outlier:
            new_array_y = np.append(new_array_y, y_values[i])
            new_array_x = np.append(new_array_x, time_values[i])
            new_array_uncerts = np.append(new_array_uncerts, uncertainties[i])
    # produces a plot of expected and measured wavelengths against time
    plt.errorbar\
        (new_array_x,new_array_y,\
         new_array_uncerts, fmt='o', label = 'measured wavelengths')
    plt.plot(xv_array, y_2_values, 'o', label = 'expected wavelengths')
    plt.xlabel('Time (s)')
    plt.ylabel('Wavelength (m)')
    plt.title('Wavelength observed (m) against time (s)', pad=20)
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

def wavelength_expected_function(v_0, time, phi, omega, lambda_0):
    """
    # calculates the expected wavelength for the given v_0, omega and 
    line of sight angle values at the given time 

    :param v_0:
    :param time:
    :param phi:
    :param omega:
    :param lambda_0:
    :return:
    """
    theta_1 = math.radians(THETA)
    c_value = 3e08
    phi = 16.83*math.pi/18
    # calculation of expected wavelength using given parameters
    wavelength_expected = ((c_value+math.sin(theta_1)*\
                            (v_0*math.sin(omega*time+phi)))/c_value)*lambda_0
    return wavelength_expected

def find_reduced_chi_squared(chi_squared, num_points, num_unkown_params):
    """
    # finds the reduced chi-squared value at the given number of data points 
    for a given number of unknown parameters 

    :param chi-squared:
    :param num_points:
    :param num_unknown_params:
    :return:
    """
    # calculates and rounds reduced chi squared with given parameters
    reduced_chi_squared = round(chi_squared/(num_points - num_unkown_params),\
                                3)
    return reduced_chi_squared

def get_v_0(minima):
    """
    # getter function for obtaining the v_0 value 
    (and rounding it) from a tuple containing v_0 and omega 

    :param minima:
    :return:
    """
    v_0 = sig_figs(minima[0], 4)
    return v_0

def get_omega(minima):
    """
    # getter function for obtaining the omega value (and rounding it) 
    from a tuple containing v_0 and omega 

    :param minima:
    :return:
    """
    omega = sig_figs(minima[1], 4)
    return omega

def sig_figs(num, precision):
    """
    # rounds the given number to the given number of significant figures 

    :param num:
    :param precision:
    :return:
    """
    num = float(num)
    return round(num, -int(floor(log10(abs(num)))) + (precision - 1))

def get_r(omega):
    """
    # calculates the distance from the star for the given omega value 

    :param omega:
    :return:
    """
    m_s = 2.78*1.989e30
    grav_constant = 6.67430e-11
    r_value = math.pow((grav_constant*m_s)/omega**2, 1/3)
    r_value = r_value*6.68459e-12
    return r_value

def get_m_p(r_au, v_0):
    """
    # calculates the mass of the planet for the given distance from 
    the star and the given v_0 value  

    :param r_AU:
    :param v_0:
    :return:
    """
    m_s = 2.78*1.989e30 # converts star's mass from Jovian masses to kg
    grav_constant = 6.67430e-11
    r_value = r_au*1.5e11 # converts r from AU to metres
    m_p = m_s*v_0*math.pow((r_value/(grav_constant*m_s)),1/2)/(1.898e27)
    return m_p

def contour_plot_function\
    (v_0_actual, omega_actual, chi_actual, m_p, r_value):
    """
    # creates a contour plot of the reduced chi-squared value for
    different v_0 and omega values  

    :param v_0_actual:
    :param omega_actual:
    :param chi_actual:
    :param mp:
    :param r:
    :return:
    """
    v_0_values = np.linspace(0.8*v_0_actual,1.1*v_0_actual, 50)
    omega_values = np.linspace(0.9*omega_actual, 1.1*omega_actual, 50)
    chi_squared_list = []
    chi_squared_total = []
    v_0_mesh, omega_mesh = np.meshgrid(v_0_values, omega_values)
    max_x = 0
    max_y = 0
    # calculates the chi-squared for every combination of v_0 and omega values
    for i in range(len(v_0_values)):
        chi_squared_list = []
        for j in range(len(omega_values)):
            chi_squared_current = find_chi_squared\
                (v_0_mesh[0,i], omega_mesh[j,0])
            chi_squared_current = find_reduced_chi_squared\
                (chi_squared_current, NUM_DATA_POINTS, 2)
            # makes a list of reduced chi-squared for the current v_0 value...
            #...with every omega value in the mesh array
            chi_squared_list.append(chi_squared_current)
        """creates a list of lists; each list corresponds to a
        specific v_0 value and contains reduced chi-squared values
        of that v_0 value and every omega value in the mesh arrays"""
        chi_squared_total.append(chi_squared_list)
    fig = plt.figure()
    ax_plot = fig.add_subplot(111)
    contour_plot = ax_plot.contourf(v_0_mesh, omega_mesh, chi_squared_total)
    fig.colorbar(contour_plot)
    ax_plot.set_xlabel('v_0 (m/s)')
    ax_plot.set_ylabel('Omega (Hz)')
    ax_plot.set_title\
        ('Reduced Chi-Squared Contour against v_0 and omega', pad=20)
    plt.savefig('contour_plot.png')
    plt.show()
    # uncertainty in v_0 and omega values when contour is reduced chi-squared+1
    desired_contour_level = round(1 + chi_actual, 2)
    tolerance = 0.05
    contour_line = None
    for level, line in zip(contour_plot.levels, contour_plot.collections):
        if np.isclose(level, desired_contour_level, atol=tolerance):
            contour_line = line
            break
    if contour_line is None: # if reduced chi-squared + 1 contour not found
        print("Desired contour level not found")
        uncertainties =\
            (False, max_x, max_y, v_0_actual, omega_actual, m_p, r_value)
    else:
        """creates arrays of v_0, omega values when 
        contour is reduced chi-squared +1"""
        x_contour = contour_line.get_paths()[0].vertices[:,0]
        y_contour = contour_line.get_paths()[0].vertices[:,1]
        """maximum uncertainty in v_0, omega found at maximum values of 
        x_contour, y_contour"""
        max_x = np.max(x_contour)
        max_y = np.max(y_contour)
        uncertainties = find_uncertainties\
            (True, max_x, max_y, v_0_actual, omega_actual, m_p, r_value)
    return uncertainties

def find_uncertainties(boolval, max_v_0, max_omega, v_0, omega, m_p, radius):
    """
    # finds the uncertainties on the optimum v_0 and omega values,
    as well as on the mass of the planet and the distance from the star

    :param boolval:
    :param max_v_0:
    :param max_omega:
    :param v_0:
    :param omega:
    :param m_p:
    :param radius:
    :return:
    """
    if boolval is True: # if uncertainties can be found from contour plot
        total_list = []
        uncert_v_0 = max_v_0 - v_0
        uncert_omega = max_omega - omega
        uncert_radius =(2/3)*(uncert_omega/omega)*radius
        uncert_mp = ((uncert_v_0/v_0)*0.5*(uncert_radius/radius)*m_p)/1.898e27
        uncert_radius = uncert_radius*6.68459e-12
        total_list.append(uncert_v_0)
        total_list.append(uncert_omega)
        total_list.append(uncert_mp)
        total_list.append(uncert_radius)
    else: # if uncertainties not found from contour plot
        total_list = [0,0,0,0,0,0]
    return total_list

def ask_theta():
    """
    # asks for the line of sight value in degrees
    
    """
    global THETA
    while True:
        try:
            THETA = float(input\
                          ("Please enter the line of sight angle in degrees:"))
            break
        except ValueError: # ensures angle entered is a float/integer
            print("Please input a valid angle")

def main():
    """
    # runs all other required functions  

    """
    ask_theta() # reads in line of sight angle
    total_list = read_file() # reads in data from files
    # unpacks time, wavelength, uncertainties, errors from data from files
    time_values = total_list[0]
    wavelength_values = total_list[1]
    uncertainties = total_list[2]
    file_errors = total_list[3]
    for i in range(len(file_errors)): # prints errors when reading file data
        print(file_errors[i])
    print("----------------------------------------------")
    values = find_values(bool_val=False, total_uncertainties=[])
    # unpacks different variables from the values list
    v_0 = values[0]
    omega = values[1]
    reduced_chi = values[2]
    r_value = values[3]*1.496e11
    m_p = values[4]*1.898e27
    # produces a plot of data
    produce_plot(time_values, wavelength_values, v_0, omega, uncertainties)
    # produces a contour plot of data & calculates uncertainties on variables
    uncertainties_total = \
        contour_plot_function(v_0, omega, reduced_chi, m_p, r_value)
    print("----------------------------------------------")
    # prints the values and their uncertainties
    values = \
        find_values(bool_val=True, total_uncertainties=uncertainties_total)

if __name__ == "__main__":
    main()
