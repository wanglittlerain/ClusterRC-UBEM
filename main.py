"""
Urban Building Energy Demand Model - Enhanced 4R1C RC model.
It is a novel reduced-order model for evaluating urban building energy demand supporting multiple thermal zone applications based on clustered resistance-capacitance (RC) network.
Cite us: Applied Energy 361 (2024): 122896.
"""

__author__ = "Xiaoyu Wang"
__copyright__ = ["Copyright 2025, College of Architecture and Urban Planning (CAUP) - Tongji University"]
__credits__ = ["Xiaoyu Wang"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = [""]
__email__ = ["wanglittlerain@163.com"]
__status__ = "Production"


import numpy as np
import pandas as pd
from pandas import read_excel
import math as m
from math import inf
import json




def rad2degree(rad):
    """Convert radians to degrees"""
    return rad * 180 / m.pi

def degree2rad(degree):
    """Convert degrees to radians"""
    return m.pi * degree / 180

def max_min(x1, x2):
    """Return the maximum of two values"""
    if x1 > x2:
        x3 = x1
    else:
        x3 = x2
    return x3


def thermalzone_body_center(ExteriorWall_collect, InternalWall_collect):
    """
    Calculate the centroid coordinates of a thermal zone
    :param ExteriorWall_collect: Collection of exterior wall surfaces
    :param InternalWall_collect: Collection of internal wall surfaces
    :return: Centroid coordinates [x,y,z]
    """
    wall_collect = []
    wall_collect.extend(ExteriorWall_collect)
    wall_collect.extend(InternalWall_collect)
    minx, maxx, miny, maxy, minz, maxz = inf, -inf, inf, -inf, inf, -inf
    for surface in wall_collect:
        for point in surface:
            x, y, z = point
            if x > maxx:
                maxx = x
            if x < minx:
                minx = x
            if y > maxy:
                maxy = y
            if y < miny:
                miny = y
            if z > maxz:
                maxz = z
            if z < minz:
                minz = z
    center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
    return center


def calc_plane_func(pa, pb, pc, center):
    """
    Calculate the plane equation
    :param pa: First point
    :param pb: Second point
    :param pc: Third point
    :param center: Thermal zone center coordinates
    :return: Plane equation as Ax+By+Cz=D, returns ((A,B,C),D)
             (A,B,C) is the normal vector pointing outward
    """
    A = np.array(pa)
    B = np.array(pb)
    C = np.array(pc)
    AB = B - A
    AC = C - A
    f = np.cross(AB, AC)
    K = np.dot(f, A)
    c = (A + C) / 2
    l = c - np.array(center)
    if np.dot(f, l) >= 0:
        return [list(f), K]
    else:
        return [list(-f), -K]

def calc_rotate_mat(plane, center):
    """
    Rotate plane to horizontal surface
    :param plane: All vertex coordinates of the plane
    :param center: Thermal zone center coordinates
    :return: Rotated vertex coordinates in horizontal plane
    """
    # Ground plane equation
    Plane_ground = calc_plane_func([0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, -1])
    plane_func = calc_plane_func(plane[0], plane[1], plane[2], center)
    i_f, i_k = plane_func
    rotate_mat = []
    # Calculate angle between this plane and XY plane (ground)
    angle = m.acos(np.dot(i_f, Plane_ground[0])/(np.linalg.norm(i_f)*np.linalg.norm(Plane_ground[0])))
    # Calculate rotation axis vector
    rv = np.cross(Plane_ground[0], i_f)
    rotate = True
    plane_rotate = []
    if np.linalg.norm(rv) == 0:
        # Parallel to ground plane, no rotation needed
        rotate = False
        for point in plane:
            plane_rotate.append([point[0], point[1]])
    if rotate:
        rv_len = np.linalg.norm(rv)
        for kk in range(len(rv)):
            # Normalize rotation vector
            rv[kk] /= rv_len
        # Rotation matrix (see: https://en.wikipedia.org/wiki/Rotation_matrix)
        cost = m.cos(angle)
        sint = m.sin(angle)
        xx = rv[0]
        yy = rv[1]
        zz = rv[2]
        rotate_mat = np.mat(
            [
                [cost+(1-cost)*(xx**2), (1-cost)*xx*yy-sint*zz, (1-cost)*xx*zz+sint*yy],
                [(1-cost)*yy*xx+sint*zz, cost+(1-cost)*(yy**2), (1-cost)*yy*zz-sint*xx],
                [(1-cost)*xx*zz-sint*yy, (1-cost)*zz*yy+sint*xx, cost+(1-cost)*(zz**2)]
            ]
        )
        # Apply inverse rotation matrix
        for point in plane:
            p = rotate_mat * np.array([[point[0]], [point[1]], [point[2]]])
            plane_rotate.append([float(p[0]), float(p[1])])
    return plane_rotate

def calc_area_2d(points):
    """
    Calculate area of a convex polygon in 2D
    :param points: Vertex coordinates of a plane surface (convex polygon)
    :return: Area of polygon
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area/2)


def thermalzone_body_volume(Roof_collect, Ground_collect, Raised_Ground_collect, Floor_collect, Ceiling_collect, center):
    """
    Calculate volume of thermal zone
    :param Roof_collect: Roof surfaces
    :param Ground_collect: Ground surfaces
    :param Raised_Ground_collect: Raised ground surfaces
    :param Floor_collect: Floor surfaces
    :param Ceiling_collect: Ceiling surfaces
    :param center: Thermal zone center coordinates
    :return: Volume of thermal zone
    """
    plane_collect = []
    plane_collect.extend(Ground_collect)
    plane_collect.extend(Floor_collect)
    plane_collect.extend(Raised_Ground_collect)
    plane_collect.extend(Ceiling_collect)
    plane_collect.extend(Roof_collect)
    minz, maxz = inf, -inf
    area = 0
    for surface in plane_collect:
        area += calc_area_2d(calc_rotate_mat(surface, center))
        for point in surface:
            x, y, z = point
            if z > maxz:
                maxz = z
            if z < minz:
                minz = z
    height = maxz - minz
    volume = area * height / 2
    return volume


def distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    x_d = point1[0] - point2[0]
    y_d = point1[1] - point2[1]
    z_d = point1[2] - point2[2]
    return (x_d**2 + y_d**2 + z_d**2) ** 0.5

def calc_perimeter(points):
    """Calculate perimeter of a polygon"""
    perimeter = 0
    for i in range(0, len(points)-1):
        pt1 = points[i]
        pt2 = points[i+1]
        perimeter += distance(pt1, pt2)
        if i + 2 == len(points):
            perimeter += distance(points[-1], points[0])
        else:
            continue
    return perimeter



def Exterior_convective_heat_transfer_coefficient(plane, roughness, v, wind_dir, T_s_o, T_out, center):
    '''
    method: TARP ALGORITHM
    Calculate exterior wall convective heat transfer coefficient (W/m²·K)
    
    :param plane: Coordinates of all vertices of the plane
    :param roughness: Surface roughness category.
                     very rough: plaster
                     rough: brick
                     medium rough: concrete
                     medium smooth: softwood
                     smooth: gypsum putty
                     very smooth: glass
    :param v: Outdoor wind speed (m/s)
    :param wind_dir: Outdoor wind direction in degrees (e.g., N = 0°, S = 180°)
    :param T_s_o: Exterior surface temperature (°C)
    :param T_out: Outdoor air temperature (°C)
    :param center: Center coordinates of the thermal zone
    :return: Convective heat transfer coefficient (U_e)
    '''
    if roughness == 'very rough':
        Rf = 2.17  # Surface roughness factor
    elif roughness == 'rough':
        Rf = 1.67
    elif roughness == 'medium rough':
        Rf = 1.52
    elif roughness == 'medium smooth':
        Rf = 1.13
    elif roughness == 'smooth':
        Rf = 1.11
    elif roughness == 'very smooth':
        Rf = 1.00
    else:
        Rf = 1.13

    vector_wind = [-m.sin(degree2rad(wind_dir)), -m.cos(degree2rad(wind_dir)), -1]  # Wind direction vector
    plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]  # Plane normal vector

    # Determine if the surface is windward
    if np.dot(vector_wind, plane_i_f) < 0:
        Wf = 1
    else:
        Wf = 0.5

    P = calc_perimeter(plane)  # Plane perimeter
    A = calc_area_2d(calc_rotate_mat(plane, center))  # Projected 2D area of the plane

    # Forced convection coefficient
    h_e_forced = 2.537 * Wf * Rf * m.sqrt(P * v / A)

    ground_i_f = [0, 0, 1]  # Ground normal vector

    # Natural convection coefficient depending on orientation and temperature difference
    if np.dot(ground_i_f, plane_i_f) == 0:
        h_e_natural = 1.31 * abs(T_s_o - T_out) ** (1 / 3)
    elif (np.dot(ground_i_f, plane_i_f) < 0) and (T_s_o - T_out) > 0:
        h_e_natural = 9.482 * abs(T_s_o - T_out) ** (1 / 3) / (7.283 - 1)
    elif (np.dot(ground_i_f, plane_i_f) < 0) and (T_s_o - T_out) < 0:
        h_e_natural = 1.81 * abs(T_s_o - T_out) ** (1 / 3) / (1.382 + 1)
    elif (np.dot(ground_i_f, plane_i_f) > 0) and (T_s_o - T_out) < 0:
        h_e_natural = 9.482 * abs(T_s_o - T_out) ** (1 / 3) / (7.283 - 1)
    elif (np.dot(ground_i_f, plane_i_f) > 0) and (T_s_o - T_out) > 0:
        h_e_natural = 1.81 * abs(T_s_o - T_out) ** (1 / 3) / (1.382 + 1)
    else:
        h_e_natural = 0.0

    U_e = h_e_forced + h_e_natural

    # Ensure non-zero result
    if U_e == 0.0:
        U_e = 1e-8

    return U_e



def Interior_convective_heat_transfer_coefficient(plane, T_s_i, T_a_in, center):
    '''
    method: TARP ALGORITHM
    Calculate interior wall convective heat transfer coefficient (W/m²·K)
    
    :param plane: Coordinates of all vertices of the plane
    :param T_s_i: Interior surface temperature (°C)
    :param T_a_in: Indoor air temperature (°C)
    :param center: Center coordinates of the thermal zone
    :return: Convective heat transfer coefficient (U_i)
    '''
    ground_i_f = [0, 0, 1]  # Ground normal vector
    plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]  # Plane normal vector

    # Determine natural convection based on orientation and temperature difference
    if np.dot(ground_i_f, plane_i_f) == 0:
        h_i_natural = 1.31 * abs(T_s_i - T_a_in) ** (1 / 3)
    elif (np.dot(ground_i_f, plane_i_f) > 0) and (T_s_i - T_a_in) > 0:
        h_i_natural = 9.482 * abs(T_s_i - T_a_in) ** (1 / 3) / (7.283 - 1)
    elif (np.dot(ground_i_f, plane_i_f) > 0) and (T_s_i - T_a_in) < 0:
        h_i_natural = 1.81 * abs(T_s_i - T_a_in) ** (1 / 3) / (1.382 + 1)
    elif (np.dot(ground_i_f, plane_i_f) < 0) and (T_s_i - T_a_in) < 0:
        h_i_natural = 9.482 * abs(T_s_i - T_a_in) ** (1 / 3) / (7.283 - 1)
    elif (np.dot(ground_i_f, plane_i_f) < 0) and (T_s_i - T_a_in) > 0:
        h_i_natural = 1.81 * abs(T_s_i - T_a_in) ** (1 / 3) / (1.382 + 1)
    else:
        h_i_natural = 0.0

    # Ensure non-zero result
    if h_i_natural == 0.0:
        h_i_natural = 1e-8

    U_i = h_i_natural
    return U_i


def calc_declination(timestep):
    '''
    Calculate solar declination angle based on the given timestep.
    
    :param timestep: Time step (hour index, e.g., 1–8760 for hourly data over a year)
    :return: Solar declination angle in radians
    '''
    n = (timestep - 1) // 24 + 1  # Day of the year
    declination = 23.45 * m.sin(degree2rad(360 * (284 + n) / 365))  # Declination in degrees
    return degree2rad(declination)  # Convert to radians

def calc_solar_angle(timestep, latitude, longitude, declination, Ls, hemicircle):
    '''
    Calculate solar altitude angle (solar elevation angle).
    
    :param timestep: Time step (hour index, e.g., 1–8760 for hourly data)
    :param latitude: Geographic latitude (in degrees)
    :param longitude: Geographic longitude (in degrees)
    :param declination: Solar declination angle (in radians)
    :param Ls: Standard meridian for the time zone (in degrees, e.g., 120 for UTC+8)
    :param hemicircle: Hemisphere location ('E' for Eastern Hemisphere, 'W' for Western Hemisphere)
    :return: Solar altitude angle (in radians)
    '''
    n = (timestep - 1) // 24 + 1  # Day of the year

    # Solar time equation (in minutes)
    W = 2 * n * m.pi / 360
    e = (-0.0002786409 
         + 0.1227715 * m.cos(W + 1.498311) 
         - 0.1654575 * m.cos(2 * W - 1.261546) 
         - 0.00535383 * m.cos(3 * W - 1.1571))

    ts = timestep % 24  # Hour of the day (0–23)

    # Hour angle (in radians)
    w = degree2rad(15 * (ts - 12 + e / 60 + (longitude - Ls) / 15))

    # Solar altitude angle (in radians)
    sin_h = (m.sin(degree2rad(latitude)) * m.sin(declination) +
             m.cos(degree2rad(latitude)) * m.cos(declination) * m.cos(w))
    
    h = m.asin(sin_h)
    return h



def calc_azimuth_angle(timestep, h, latitude, longitude, declination, Ls, hemicircle):
    '''
    Calculate the solar azimuth angle.
    
    :param timestep: Time step (hour index, e.g., 1–8760 for hourly data)
    :param h: Solar altitude angle (in radians)
    :param latitude: Geographic latitude (in degrees)
    :param longitude: Geographic longitude (in degrees)
    :param declination: Solar declination angle (in radians)
    :param Ls: Standard meridian for the time zone (in degrees, e.g., 120 for UTC+8)
    :param hemicircle: Hemisphere indicator ('E' for Eastern Hemisphere, 'W' for Western Hemisphere) – not used in current implementation
    :return: Solar azimuth angle `a` (in radians). Positive is west of south, negative is east of south.
    '''
    n = (timestep - 1) // 24 + 1  # Day of the year

    # Equation of time correction (in minutes)
    W = 2 * n * m.pi / 360
    e = (-0.0002786409
         + 0.1227715 * m.cos(W + 1.498311)
         - 0.1654575 * m.cos(2 * W - 1.261546)
         - 0.00535383 * m.cos(3 * W - 1.1571))

    ts = timestep % 24  # Hour of the day (0–23)

    # Hour angle (in radians)
    w = degree2rad(15 * (ts - 12 + e / 60 + (longitude - Ls) / 15))

    # If the sun is directly south (solar noon)
    if w == 0:
        return 0
    else:
        # Calculate azimuth angle using spherical trigonometry
        cos_a = ((m.sin(h) * m.sin(degree2rad(latitude)) - m.sin(declination)) /
                 (m.cos(h) * m.cos(degree2rad(latitude))))
        a = m.acos(cos_a)

        # Adjust sign based on morning (east) or afternoon (west)
        if w < 0:
            a = -a

    return a




def Direct_Radiation(h, a, DNR, plane, center):
    '''
    Compute the direct solar radiation received on a surface (W/m²).
    
    :param h: Solar altitude angle (in radians)
    :param a: Solar azimuth angle (in radians)
    :param DNR: Direct Normal Radiation (W/m²)
    :param plane: List of vertex coordinates defining the surface
    :param center: Center coordinates of the thermal zone
    :return: Tuple (DirectRadiation_wall, DirectRadiation_window, cos_i)
             - DirectRadiation_wall: direct radiation received by opaque surfaces (e.g. walls)
             - DirectRadiation_window: direct radiation received by transparent surfaces (e.g. windows)
             - cos_i: cosine of the incident angle between sun vector and surface normal
    '''
    if h <= 0:
        # Sun is below the horizon
        DirectRadiation_wall = 0
        DirectRadiation_window = 0
        cos_i = 0
    else:
        # Compute surface normal vector
        plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]
        
        # Solar incidence vector
        solar_vector = [m.cos(h) * m.sin(a), m.cos(h) * m.cos(a), -m.sin(h)]
        
        # Cosine of incidence angle between surface normal and sun vector
        cos_i = np.dot(plane_i_f, solar_vector) / (np.linalg.norm(plane_i_f) * np.linalg.norm(solar_vector))
        
        if cos_i >= 0:
            # Backside surface or shaded area
            sunlit_ratio_wall = 0    # Ratio of sunlit area for opaque surfaces (1 - shading factor)
            sunlit_ratio_window = 0  # Ratio of sunlit area for transparent surfaces (1 - shading factor)
            cos_i = 0
        else:
            # Front-facing, sunlit surface
            sunlit_ratio_wall = 1
            sunlit_ratio_window = 1
            cos_i = abs(cos_i)
        
        DirectRadiation_wall = DNR * cos_i * sunlit_ratio_wall
        DirectRadiation_window = DNR * cos_i * sunlit_ratio_window

    return DirectRadiation_wall, DirectRadiation_window, cos_i



def Longwave_Radiation(T_s_o, T_a_out, plane, emittance, T_dp, OSC, center):
    '''
    Compute the net longwave radiation exchange on a surface (W/m²).
    
    :param T_s_o: Exterior surface temperature (°C)
    :param T_a_out: Outdoor air temperature (°C)
    :param plane: List of vertex coordinates defining the surface
    :param emittance: Longwave emissivity of the surface (dimensionless)
    :param T_dp: Dew point temperature (°C)
    :param OSC: Opaque sky cover (tenths, range 0–10)
    :param center: Center coordinates of the thermal zone
    :return: Longwave radiation heat exchange (W/m²)
    '''
    # Sky emissivity based on dew point and sky cover
    emit_sky = ((0.787 + 0.764 * m.log((273.15 + T_dp) / 273)) *
                (1 + 0.0224 * OSC - 0.0035 * OSC**2 + 0.00028 * OSC**3))
    
    # Surface normal vector
    plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]
    ground_i_f = [0, 0, 1]  # Ground normal vector
    
    # Angle between surface and horizontal
    cos_sita = np.dot(plane_i_f, ground_i_f) / (np.linalg.norm(plane_i_f) * np.linalg.norm(ground_i_f))
    
    # View factors
    GVF = 0.5 * (1 - cos_sita)  # Ground view factor
    SVF = 0.5 * (1 + cos_sita)  # Sky view factor
    
    beta = m.sqrt(SVF)  # Cloud attenuation factor for sky
    
    # Longwave radiation exchange: sky + ground + ambient reflection
    Longwave_Radiation = (
        emittance * 5.67e-8 * SVF * beta * (emit_sky * (T_a_out + 273.15)**4 - (T_s_o + 273.15)**4) +
        emittance * 5.67e-8 * GVF * ((T_a_out + 273.15)**4 - (T_s_o + 273.15)**4) +
        emittance * 5.67e-8 * SVF * (1 - beta) * ((T_a_out + 273.15)**4 - (T_s_o + 273.15)**4)
    )
    
    return Longwave_Radiation



def Diffuse_Radiation(DHR, plane, DNR, h, a, center):
    '''
    Perez model for calculating diffuse solar radiation on a tilted surface.
    
    :param DHR: Diffuse Horizontal Radiation (W/m²)
    :param plane: Vertex coordinates of the surface plane
    :param DNR: Direct Normal Radiation (W/m²)
    :param h: Solar altitude angle (radians)
    :param a: Solar azimuth angle (radians)
    :param center: Center coordinate of the thermal zone
    :return: Diffuse radiation on the inclined surface (W/m²)
    '''
    if (h <= 0) or (DHR == 0):
        DiffuseRadiation = 0
    else:
        plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]  # Plane normal vector
        solar_vector = [m.cos(h) * m.sin(a), m.cos(h) * m.cos(a), -m.sin(h)]
        cos_i = -np.dot(plane_i_f, solar_vector)/(np.linalg.norm(plane_i_f)*np.linalg.norm(solar_vector))
        ground_i_f = [0, 0, 1]
        cos_sita = np.dot(plane_i_f, ground_i_f)/(np.linalg.norm(plane_i_f)*np.linalg.norm(ground_i_f))
        sin_sita = m.sqrt(1 - cos_sita**2)
        z = m.pi/2 - h  # Zenith angle
        coef_a = max_min(0, cos_i)
        coef_b = max_min(0.087, m.cos(z))

        # Sky clearness index
        emothno = ((DHR + DNR) / DHR + 1.041 * z**3) / (1 + 1.041 * z**3)

        # Select Perez coefficients based on clearness index
        if 1 <= emothno < 1.065:
            F11, F12, F13 = -0.0083117, 0.5877285, -0.0620636
            F21, F22, F23 = -0.0596012, 0.0721249, -0.0220216
        elif 1.065 <= emothno < 1.23:
            F11, F12, F13 = 0.1299457, 0.6825954, -0.1513752
            F21, F22, F23 = -0.0189325, 0.0659650, -0.0288748
        elif 1.23 <= emothno < 1.5:
            F11, F12, F13 = 0.3296958, 0.4868735, -0.2210958
            F21, F22, F23 = 0.0554140, -0.0639588, -0.0260542
        elif 1.5 <= emothno < 1.95:
            F11, F12, F13 = 0.5682053, 0.1874525, -0.295129
            F21, F22, F23 = 0.1088631, -0.1519229, -0.0139754
        elif 1.95 <= emothno < 2.8:
            F11, F12, F13 = 0.873028, -0.3920403, -0.3616149
            F21, F22, F23 = 0.2255647, -0.4620442, 0.0012448
        elif 2.8 <= emothno < 4.5:
            F11, F12, F13 = 1.1326077, -1.2367284, -0.4118494
            F21, F22, F23 = 0.2877813, -0.8230357, 0.0558651
        elif 4.5 <= emothno < 6.2:
            F11, F12, F13 = 1.0601591, -1.5999137, -0.3589221
            F21, F22, F23 = 0.2642124, -1.127234, 0.1310694
        else:
            F11, F12, F13 = 0.677747, -0.3272588, -0.2504286
            F21, F22, F23 = 0.1561313, -1.3765031, 0.2506212

        # Brightness index (air mass factor)
        derta = DHR / (m.sin(h) + 0.15 * (3.885 + h)**(-1.253)) / 1353
        F1 = F11 + F12 * derta + F13 * z
        F2 = F21 + F22 * derta + F23 * z

        # Final Perez model equation
        DiffuseRadiation = (
            DHR * F2 * sin_sita +
            DHR * (1 - F1) * (1 + cos_sita) / 2 +
            DHR * F1 * coef_a / coef_b
        )
    return DiffuseRadiation


def ground_reflect_radiation(plane, DHR, DNR, h, ground_reflectance, center):
    '''
    ground_reflect_radiation: W/m2
    :param plane: List of plane vertex coordinates
    :param DHR: Diffuse Horizontal Radiation W/m2
    :param DNR: Direct Normal Radiation
    :param h: Solar altitude angle
    :param ground_reflectance: Ground reflectance
    :param center: Center coordinate of the thermal zone
    :return:
    '''
    plane_i_f = calc_plane_func(plane[0], plane[1], plane[2], center)[0]  # Normal vector of the plane
    ground_i_f = [0, 0, 1]
    cos_sita = np.dot(plane_i_f, ground_i_f) / (np.linalg.norm(plane_i_f) * np.linalg.norm(ground_i_f))
    GVF = 0.5 * (1 - cos_sita) * Rdome_G
    if h <= 0:
        GroundReflectedSolar = 0
    else:
        GroundReflectedSolar = (DNR * m.cos(m.pi / 2 - h) + DHR) * ground_reflectance * GVF
    return GroundReflectedSolar




def Uw1_Uw2(thickness_list, lamda_list):
    # Calculate equivalent thermal transmittance for the inner and outer parts of the envelope
    # (the layer closest to the indoor side is the inner part, the rest is the outer part)
    '''
    :param thickness_list: [x1, x2, x3, ...] list of thicknesses for each material layer (from inside to outside)
    :param lamda_list: [x1, x2, x3, ...] list of thermal conductivities for each layer (from inside to outside)
    :return: Uw1: Equivalent thermal transmittance of the outer part
    :return: Uw2: Equivalent thermal transmittance of the inner part
    '''
    Uw2 = lamda_list[0]/thickness_list[0]
    R_total_w1 = 0
    for i in range(1, len(thickness_list)):
        R_total_w1 += thickness_list[i] / lamda_list[i]
    Uw1 = 1 / R_total_w1
    return Uw1, Uw2


def Uw_total(thickness_list, lamda_list):
    # Calculate overall equivalent thermal transmittance of the envelope
    '''
    :param thickness_list: [x1, x2, x3, ...] list of thicknesses for each material layer (from inside to outside)
    :param lamda_list: [x1, x2, x3, ...] list of thermal conductivities for each layer (from inside to outside)
    :return: Uw: Overall equivalent thermal transmittance
    '''
    R_total_w = 0
    for i in range(0, len(thickness_list)):
        R_total_w += thickness_list[i] / lamda_list[i]
    Uw = 1 / R_total_w
    return Uw


def Cw_coef(thickness_list, rio_list, Cp_list):
    # Calculate heat capacity coefficient of the wall
    Cw_coef = 0
    for i in range(0, len(thickness_list)):
        Cw_coef += thickness_list[i]*rio_list[i]*Cp_list[i]
    return Cw_coef


def Deep_ground_temp(Out_DryTemp_All):
    # Calculate average deep ground temperature from annual outdoor dry-bulb temperatures
    temp_ground = 0
    for temp in Out_DryTemp_All:
        temp_ground += temp
    temp_ground_average = temp_ground / 8760
    return temp_ground_average


def Uw1_sgf(ground_coordinates, d_ew, R_fg, Ui_fg):
    """
    :param ground_coordinates: List of ground surface coordinates
    :param d_ew: External wall thickness
    :param R_fg: Total thermal resistance of ground material layers
    :return: Equivalent thermal transmittance of the soil layer (W/m2·K)
    """
    A_fg = calc_area_2d(ground_coordinates)
    P_fg = calc_perimeter(ground_coordinates)
    B = A_fg / (0.5 * P_fg)
    df = d_ew + 2 * (1 / Ui_fg + R_fg + 0.03)
    if df < B:
        U_fg = 2 * 2 / (m.pi * B + df) * m.log(m.pi * B / df + 1)
    else:
        U_fg = 2 / (0.457 * B + df)
    Uw1_ground = 1 / (1 / U_fg - 1 / Ui_fg - R_fg)
    return Uw1_ground



def read_geo_json(file):
    '''
    Read geo_json file.
    Thermal zone data format:
    --BuildingID: number
    --ThermalZoneID: number
    --ExteriorWall: [[ExteriorWall1], [ExteriorWall2], [ExteriorWall3], [ExteriorWall4], [...]]
    --Roof: [[Roof1], [Roof2], [Roof3], [Roof4], [...]]
    --Ground: [[Ground1], [Ground2], [Ground3], [Ground4], [...]]
    --Raised_Ground: [[Raised_Ground1], [Raised_Ground2], [Raised_Ground3], [Raised_Ground4], [...]]
    --InternalWall: [[[InternalWall1], Connected_ThermalZoneID_number], [[InternalWall2], Connected_ThermalZoneID_number], [[...], ...]]
    --InternalMass: [[InternalMass1], [InternalMass2], [InternalMass3], [InternalMass4], [...]]
    --Floor: [[[Floor1], Connected_ThermalZoneID_number], [[Floor2], Connected_ThermalZoneID_number], [[...], ...]]
    --Ceiling: [[[Ceiling1], Connected_ThermalZoneID_number], [[Ceiling2], Connected_ThermalZoneID_number], [[...], ...]]
    Note: All surfaces must be convex polygons. If concave, please split them into sets of convex polygons.
    
    :param file: path to the geo_json file
    :return:
    n: number of thermal zones
    Returns a series of surface collections for each thermal zone
    '''
    with open(file, 'r') as f:
        f_data = f.read()
        try:
            data = json.loads(f_data)
            n = len(data)
            ExteriorWall_ThermalZone_Collect = [0] * n
            Roof_ThermalZone_Collect = [0] * n
            Ground_ThermalZone_Collect = [0] * n
            Raised_Ground_ThermalZone_Collect = [0] * n
            InternalWall_ThermalZone_Collect = [0] * n
            InternalMass_ThermalZone_Collect = [0] * n
            Floor_ThermalZone_Collect = [0] * n
            Ceiling_ThermalZone_Collect = [0] * n
            for d in data:
                Building_ID = d.get('Building_ID')
                ThermalZoneID = d.get('ThermalZone_ID')
                ExteriorWall = d.get('ExteriorWall')
                InternalWall = d.get('InternalWall')
                InternalMass = d.get('InternalMass')
                Roof = d.get('Roof')
                Ground = d.get('Ground')
                Raised_Ground = d.get('Raised_Ground')
                Floor = d.get('Floor')
                Ceiling = d.get('Ceiling')
                ExteriorWall_ThermalZone_Collect[ThermalZoneID-1] = ExteriorWall
                Roof_ThermalZone_Collect[ThermalZoneID-1] = Roof
                Ground_ThermalZone_Collect[ThermalZoneID-1] = Ground
                Raised_Ground_ThermalZone_Collect[ThermalZoneID-1] = Raised_Ground
                InternalWall_ThermalZone_Collect[ThermalZoneID-1] = InternalWall
                InternalMass_ThermalZone_Collect[ThermalZoneID-1] = InternalMass
                Floor_ThermalZone_Collect[ThermalZoneID-1] = Floor
                Ceiling_ThermalZone_Collect[ThermalZoneID-1] = Ceiling
        except Exception as e:
            print(e)
    return n, ExteriorWall_ThermalZone_Collect, Roof_ThermalZone_Collect, Ground_ThermalZone_Collect, Raised_Ground_ThermalZone_Collect, InternalWall_ThermalZone_Collect, InternalMass_ThermalZone_Collect, Floor_ThermalZone_Collect, Ceiling_ThermalZone_Collect

def LU(A):
    '''
    Perform LU decomposition using Doolittle algorithm without pivoting.
    :param A: input square matrix
    :return: L (lower triangular), U (upper triangular)
    '''
    U = np.copy(A)
    m, n = A.shape
    L = np.eye(n)
    for k in range(n-1):
        for j in range(k+1, n):
            L[j, k] = U[j, k]/U[k, k]
            U[j, k:n] -= L[j, k] * U[k, k:n]
    return L, U





if __name__ == "__main__":

    dt = 3600  # Time step (seconds)
    nstep = int(31536000.0 / dt)  # Number of simulation steps per year


    # Define latitude and longitude (automatically generated based on user-selected city; corresponds to frontend input)
    latitude = 34.447  # XIAN
    longitude = 108.752  # XIAN
    Ls = 120  # Standard meridian for the city (°)
    site_hemicircle = 'E'  # Hemisphere of the city location: W for west, E for east


    # Import weather data (Typical Meteorological Year database for different cities; data loaded based on user-selected city; corresponds to frontend weather file)
    Typ_Meteo_data = read_excel('D:\CodingStudio\Doctoral Research\Shadow calculation\Xian_TMY.xlsx', 'Sheet1')
    Out_DryTemp = Typ_Meteo_data.iloc[:, 1]  # Outdoor dry-bulb temperature (°C)
    Out_RH = Typ_Meteo_data.iloc[:, 2] / 100  # Outdoor relative humidity
    DNR = Typ_Meteo_data.iloc[:, 3]  # Direct Normal Radiation (W/m²)
    DHR = Typ_Meteo_data.iloc[:, 4]  # Diffuse Horizontal Radiation (W/m²)
    Wind_Dir = Typ_Meteo_data.iloc[:, 5]  # Wind direction (°), N=0°, S=180°
    Wind_Speed = Typ_Meteo_data.iloc[:, 6]  # Wind speed (m/s)
    Opaque_sky_cover = Typ_Meteo_data.iloc[:, 7] / 10  # Opaque sky cover (tenths)
    Dew_point_Temp = Typ_Meteo_data.iloc[:, 8]  # Dew point temperature (°C)

    Deep_GroundTemp = Deep_ground_temp(Out_DryTemp)


    # Occupancy, mechanical ventilation, and air conditioning data (passed from frontend as hourly arrays for 8760 hours; here loaded from file as an example. 
    # In practice, each thermal zone may have its own data; here we assume all zones use the same profile.)
    people_mec_aircondition_data = read_excel(
        'D:\CodingStudio\Doctoral Research\Shadow calculation\people_mec_aircondition.xlsx', 'Sheet1')
    people_inroom_rate_allyear = people_mec_aircondition_data.iloc[:, 1]  # Indoor occupancy rate
    mec_vent_status_allyear = people_mec_aircondition_data.iloc[:, 2]  # Mechanical ventilation status
    AC_status_allyear = people_mec_aircondition_data.iloc[:, 3]  # Air conditioning status
    Eqip_use_rate_allyear = people_mec_aircondition_data.iloc[:, 4]  # Equipment usage rate
    Light_use_rate_allyear = people_mec_aircondition_data.iloc[:, 5]  # Lighting usage rate
    Indoor_set_temperature = people_mec_aircondition_data.iloc[:, 6]  # Indoor temperature setpoint (°C)




    # Input: surface collections of all thermal zones for each building (from frontend geojson)
    nZones, ExteriorWall_ThermalZones, Roof_ThermalZones, Ground_ThermalZones, Raised_Ground_ThermalZones, InternalWall_ThermalZones, InternalMass_ThermalZones, Floor_ThermalZones, Ceiling_ThermalZones = read_geo_json('Geo_MultiZone_Demo_Xian.json')

    # Window-to-wall ratios (from frontend input)
    wwr_Roof = 0.0  # Roof window-to-wall ratio
    wwr_wall_north = 0.6498  # North wall WWR
    wwr_wall_south = 0.6498  # South wall WWR
    wwr_wall_east = 0.495  # East wall WWR
    wwr_wall_west = 0.495  # West wall WWR


    # Roof material properties (from frontend selection) — CASEXIANDASHA
    thickness_roof = [0.0015, 0.2105, 0.0095]  # Thickness of each layer (inner to outer) in meters
    lamda_roof = [45.006, 0.049, 0.16]  # Thermal conductivity (W/m·K)
    rio_roof = [7680, 265, 1121.29]  # Density (kg/m³)
    Cp_roof = [418.4, 836.8, 1460]  # Specific heat capacity (J/kg·K)
    Abs_roof = [0.6, 0.7, 0.7]  # Solar absorptance
    roof_roughness = ['medium smooth', 'medium rough', 'very rough']  # Surface roughness
    Emittance_roof = [0.9, 0.9, 0.9]  # Longwave emissivity

    # Calculate roof U-values and thermal capacity
    Uw1_roof, Uw2_roof = Uw1_Uw2(thickness_roof, lamda_roof)
    Cw_roof_coef = Cw_coef(thickness_roof, rio_roof, Cp_roof)


    # Exterior wall material properties — caseXIAN
    thickness_exteriorwall = [0.0127, 0.0794, 0.2033, 0.0253]  # Thickness (m)
    lamda_exteriorwall = [0.16, 0.0432, 1.7296, 0.6918]  # Conductivity (W/m·K)
    rio_exteriorwall = [784.9, 91, 2243, 1858]  # Density (kg/m³)
    Cp_exteriorwall = [830, 837, 837, 837]  # Specific heat (J/kg·K)
    Abs_exteriorwall = [0.4, 0.5, 0.65, 0.92]  # Solar absorptance
    exteriorwall_roughness = ['smooth', 'medium rough', 'medium rough', 'smooth']  # Roughness
    Emittance_exteriorwall = [0.9, 0.9, 0.9, 0.9]  # Emissivity

    Uw1_exteriorwall, Uw2_exteriorwall = Uw1_Uw2(thickness_exteriorwall, lamda_exteriorwall)
    Cw_exteriorwall_coef = Cw_coef(thickness_exteriorwall, rio_exteriorwall, Cp_exteriorwall)

    # Total thickness of exterior wall
    d_ew = sum(thickness_exteriorwall)


    # Ground slab properties — case600
    thickness_ground = [0.1, 0.1016]
    lamda_ground = [1, 1.311]
    rio_ground = [1e-8, 2240]
    Cp_ground = [1e-8, 836.8]
    Emittance_ground = [0.9, 0.9]

    Uw2_ground = Uw_total(thickness_ground, lamda_ground)
    Cw_ground_coef = Cw_coef(thickness_ground, rio_ground, Cp_ground)


    # Raised ground properties — case980
    thickness_raised_ground = [0.1, 0.1016]
    lamda_raised_ground = [1, 1.311]
    rio_raised_ground = [1e-8, 2240]
    Cp_raised_ground = [1e-8, 836.8]
    raised_ground_roughness = ['smooth', 'rough']
    Emittance_raised_ground = [0.9, 0.9]

    Uw1_raised_ground, Uw2_raised_ground = Uw1_Uw2(thickness_raised_ground, lamda_raised_ground)
    Cw_raised_ground_coef = Cw_coef(thickness_raised_ground, rio_raised_ground, Cp_raised_ground)


    # Internal wall properties
    thickness_internalwall = [0.019, 0.003, 0.019]
    lamda_internalwall = [0.16, 0.02, 0.16]
    rio_internalwall = [800, 1.29, 800]
    Cp_internalwall = [1090, 1005, 1090]
    Emittance_internalwall = [0.9, 0.9, 0.9]

    Uw1_internalwall, Uw2_internalwall = Uw1_Uw2(thickness_internalwall, lamda_internalwall)
    Cw_internalwall_coef = Cw_coef(thickness_internalwall, rio_internalwall, Cp_internalwall)


    # Floor slab properties
    thickness_floor = [0.1016, 0.0036, 0.0191]
    lamda_floor = [0.53, 0.02, 0.06]
    rio_floor = [1280, 1.29, 368]
    Cp_floor = [840, 1005, 590]
    Emittance_floor = [0.9, 0.9, 0.9]

    Uw1_floor, Uw2_floor = Uw1_Uw2(thickness_floor, lamda_floor)
    Cw_floor_coef = Cw_coef(thickness_floor, rio_floor, Cp_floor)


    # Ceiling properties
    thickness_ceiling = [0.0191, 0.0036, 0.1016]
    lamda_ceiling = [0.06, 0.02, 0.53]
    rio_ceiling = [368, 1.29, 1280]
    Cp_ceiling = [590, 1005, 840]
    Emittance_ceiling = [0.9, 0.9, 0.9]

    Uw1_ceiling, Uw2_ceiling = Uw1_Uw2(thickness_ceiling, lamda_ceiling)
    Cw_ceiling_coef = Cw_coef(thickness_ceiling, rio_ceiling, Cp_ceiling)


    # Internal thermal mass properties
    thickness_internalmass = [0.0254]
    lamda_internalmass = [0.15]
    rio_internalmass = [608]
    Cp_internalmass = [1630]
    Emittance_internalmass = [0.9]

    Cw_internalmass_coef = Cw_coef(thickness_internalmass, rio_internalmass, Cp_internalmass)


    # Window thermal properties (from frontend)
    U_window = 2.559  # Window U-value (W/m²·K)
    # SHGC = 0.59  # Solar Heat Gain Coefficient (can be a value or a function of angle of incidence)


    # Air properties
    Rioma = 1.29  # Air density (kg/m³)
    Cpma = 1005   # Air specific heat capacity (J/kg·K)



    # Assign window-to-wall ratios (WWR) to each exterior wall surface based on its normal direction.
    # For roof surfaces, WWR is always set to wwr_Roof, regardless of segmentation.
    North_Dir = [0, 1, 0]
    East_Dir = [1, 0, 0]
    South_Dir = [0, -1, 0]
    West_Dir = [-1, 0, 0]

    # Initialize state vectors and matrices
    T = np.zeros((11 * nZones, 1))  # Current temperature state vector (for all 11 nodes per zone)
    T0 = np.zeros((11 * nZones, 1))  # Previous timestep temperature state vector
    A = np.zeros((11 * nZones, 11 * nZones))  # System coefficient matrix
    b = np.zeros((11 * nZones, 1))  # Right-hand side vector
    temp = np.zeros((11 * nZones, 1))  # Temporary storage

    # Initialize energy demand and consumption arrays
    Energy_Heat = np.zeros((nZones, 1))  # Annual heating energy per zone
    Energy_Cooling = np.zeros((nZones, 1))  # Annual cooling energy per zone
    Energy_Heat_Perarea = np.zeros((nZones, 1))  # Annual heating energy per unit area
    Energy_Cooling_Perarea = np.zeros((nZones, 1))  # Annual cooling energy per unit area
    Energy_Loads = np.zeros((2 * nZones, 1))  # Combined heating and cooling loads
    Energy_Consumption = np.zeros((2 * nZones, 1))  # Final HVAC energy consumption (includes COP)
    Energy_Consumption_Perarea = np.zeros((2 * nZones, 1))  # HVAC energy consumption per unit area

    # Coefficient of performance (COP) for HVAC systems — from frontend input
    COP_Heat = 1      # Heating COP
    COP_Cooling = 1   # Cooling COP

    # Calculate center point of each thermal zone for radiation and convection calculations
    center = []
    for n in range(nZones):
        ExteriorWall_coordinates = ExteriorWall_ThermalZones[n]
        InternalWall_coordinates = InternalWall_ThermalZones[n]
        InternalWall_collect = []
        for InternalWalls in InternalWall_coordinates:
            InternalWall_collect.append(InternalWalls[0])  # Extract wall face geometry
        center.append(thermalzone_body_center(ExteriorWall_coordinates, InternalWall_collect))

    # Initialize simulation time
    time = 0.0

    # Initialize temperature of all nodes to 5°C
    for i in range(len(T)):
        T[i] = 5




     # Start hourly simulation from January 1st 00:00 (assume Jan 1 is Monday)
    for istep in range(1, nstep + 1):

        T_new = np.zeros((11 * nZones, 1))       # New temperature vector
        Tain_set = np.zeros((11 * nZones, 1))    # Indoor temperature setpoint

        T0 = T.copy()  # Save previous timestep's temperatures

        # --- Environmental input at current hour ---
        Ta_out = Out_DryTemp[istep - 1]  # Outdoor dry-bulb temperature [°C]
        Direct_Normal_Radiation = DNR[istep - 1]  # Direct normal solar radiation [W/m²]
        Diffuse_Horizontal_Radiation = DHR[istep - 1]  # Diffuse horizontal radiation [W/m²]
        v_dir = Wind_Dir[istep - 1]  # Wind direction [deg]
        v_wind = Wind_Speed[istep - 1]  # Wind speed [m/s]
        OpqCld = Opaque_sky_cover[istep - 1]  # Opaque sky cover [tenths]
        T_dew = Dew_point_Temp[istep - 1]  # Dew point temperature [°C]

        # --- Occupancy and control schedules ---
        people_inroom_rate = people_inroom_rate_allyear[istep - 1]  # Occupancy ratio
        mec_vent_status = mec_vent_status_allyear[istep - 1]  # Mechanical ventilation on/off
        AC_status = AC_status_allyear[istep - 1]  # Air conditioning on/off
        Eqip_use_rate = Eqip_use_rate_allyear[istep - 1]  # Equipment usage rate
        Light_use_rate = Light_use_rate_allyear[istep - 1]  # Lighting usage rate
        T_indoor_set = Indoor_set_temperature[istep - 1]  # Indoor setpoint temperature [°C]

        # --- Solar position and angles ---
        declination = calc_declination(istep)  # Solar declination angle
        h = calc_solar_angle(istep, latitude, longitude, declination, Ls, site_hemicircle)  # Solar altitude
        a = calc_azimuth_angle(istep, h, latitude, longitude, declination, Ls, site_hemicircle)  # Solar azimuth

        ground_reflectance = 0.2  # Ground reflectance for solar radiation (typical for bare soil/concrete)

        # Initialize thermal/solar/area properties for all zones
        Qsun2_zones_collect = []  # Solar gain through windows
        Sr_zones_collect = []
        Qir_itc_zones_collect = []  # Internal longwave radiation
        A_czs_zones_collect = []  # Solar collecting area
        CZS_zones_collect = []
        A_total_zones_collect = []  # Total envelope area
        A_itc_area_zones_collect = []  # Internal thermal capacity area
        A_floor_area_zones_collect = []  # Floor area

        for n in range(nZones):
            # Geometry inputs for zone n
            ExteriorWall_coordinates = ExteriorWall_ThermalZones[n]
            Roof_coordinates = Roof_ThermalZones[n]
            Ground_coordinates = Ground_ThermalZones[n]
            Raised_Ground_coordinates = Raised_Ground_ThermalZones[n]
            InternalWall_coordinates = InternalWall_ThermalZones[n]
            InternalMass_coordinates = InternalMass_ThermalZones[n]
            Floor_coordinates = Floor_ThermalZones[n]
            Ceiling_coordinates = Ceiling_ThermalZones[n]

            # Initialize wall-related thermal properties
            Kw1_exteriorwall = 0     # External wall conductive gain (outside part)
            Kw2_exteriorwall = 0     # Internal part
            Cw_exteriorwall = 0      # Thermal capacity of wall
            Ke_exteriorwall = 0      # External convective conductance
            Ki_exteriorwall = 0      # Internal convective conductance
            UA_exteriorwall = 0      # Window heat transfer area
            Qsun1_exteriorwall = 0   # Solar gain on opaque wall
            Qsun2_exteriorwall = 0   # Solar gain through windows
            Qir_e_exteriorwall = 0   # Longwave radiation exchange with outdoors
            A_exteriorwall_opaque = 0
            A_exteriorwall_window = 0

            for ExteriorWall_coordinate in ExteriorWall_coordinates:
                wall_i_f = calc_plane_func(
                    ExteriorWall_coordinate[0],
                    ExteriorWall_coordinate[1],
                    ExteriorWall_coordinate[2],
                    center[n]
                )[0]  # Surface normal vector of wall

                # Determine WWR (window-to-wall ratio) by orientation
                if m.sqrt(2) / 2 <= np.dot(wall_i_f, North_Dir) / (np.linalg.norm(wall_i_f) * np.linalg.norm(North_Dir)) <= 1:
                    wwr = wwr_wall_north
                elif m.sqrt(2) / 2 <= np.dot(wall_i_f, East_Dir) / (np.linalg.norm(wall_i_f) * np.linalg.norm(East_Dir)) <= 1:
                    wwr = wwr_wall_east
                elif m.sqrt(2) / 2 <= np.dot(wall_i_f, South_Dir) / (np.linalg.norm(wall_i_f) * np.linalg.norm(South_Dir)) <= 1:
                    wwr = wwr_wall_south
                elif m.sqrt(2) / 2 <= np.dot(wall_i_f, West_Dir) / (np.linalg.norm(wall_i_f) * np.linalg.norm(West_Dir)) <= 1:
                    wwr = wwr_wall_west
                else:
                    wwr = 0  # Unknown or inclined walls

                A_wall = calc_area_2d(calc_rotate_mat(ExteriorWall_coordinate, center[n]))
                A_exteriorwall_opaque += A_wall * (1 - wwr)
                A_exteriorwall_window += A_wall * wwr

                Kw1_exteriorwall += Uw1_exteriorwall * A_wall * (1 - wwr)
                Kw2_exteriorwall += Uw2_exteriorwall * A_wall * (1 - wwr)
                Cw_exteriorwall += Cw_exteriorwall_coef * A_wall * (1 - wwr)

                # Convective heat transfer coefficients
                Ue_exteriorwall = Exterior_convective_heat_transfer_coefficient(
                    ExteriorWall_coordinate, exteriorwall_roughness[-1], v_wind, v_dir, float(T0[11 * n + 3]), Ta_out, center[n]
                )
                Ui_exteriorwall = Interior_convective_heat_transfer_coefficient(
                    ExteriorWall_coordinate, float(T0[11 * n + 1]), float(T0[11 * n]), center[n]
                )

                Ke_exteriorwall += Ue_exteriorwall * A_wall * (1 - wwr)
                Ki_exteriorwall += Ui_exteriorwall * A_wall * (1 - wwr)
                UA_exteriorwall += U_window * A_wall * wwr

                # --- Solar radiation calculations ---
                Wall_Direct_Radiation, Window_Direct_Radiation, cosi_wall = Direct_Radiation(
                    h, a, Direct_Normal_Radiation, ExteriorWall_coordinate, center[n]
                )

                SHGC_wall = 0.352  # Or use dynamic SHGC model as alternative：SHGC_wall = cosi_wall / (0.735044762230282 * cosi_wall ** 2 - 0.03408181475031143 * cosi_wall + 0.6089337726960152)

                Wall_Diffuse_Radiation = Diffuse_Radiation(
                    Diffuse_Horizontal_Radiation, ExteriorWall_coordinate, Direct_Normal_Radiation, h, a, center[n]
                )

                Wall_GroundReflect_Radiation = ground_reflect_radiation(
                    ExteriorWall_coordinate, Diffuse_Horizontal_Radiation, Direct_Normal_Radiation, h, ground_reflectance, center[n]
                )

                Wall_Total_Solar_Radiation_intensity = (
                    Wall_Direct_Radiation + Wall_Diffuse_Radiation + Wall_GroundReflect_Radiation
                )
                Wall_Window_Total_Solar_Radiation_intensity = (
                    Window_Direct_Radiation + Wall_Diffuse_Radiation + Wall_GroundReflect_Radiation
                )

                # Accumulate absorbed solar gains
                Qsun1_exteriorwall += (
                    Wall_Total_Solar_Radiation_intensity * A_wall * (1 - wwr) * Abs_exteriorwall[-1]
                )
                Qsun2_exteriorwall += (
                    Wall_Window_Total_Solar_Radiation_intensity * SHGC_wall * A_wall * wwr
                )

                # Longwave radiation loss/gain with environment
                Qir_e_exteriorwall += Longwave_Radiation(
                    float(T0[11 * n + 3]), Ta_out, ExteriorWall_coordinate,
                    Emittance_exteriorwall[-1], T_dew, OpqCld, center[n]
                ) * A_wall * (1 - wwr)




            Kw1_roof = 0
            Kw2_roof = 0
            Cw_roof = 0
            Ke_roof = 0
            Ki_roof = 0
            UA_roof = 0
            Qsun1_roof = 0
            Qsun2_roof = 0
            Qir_e_roof = 0
            A_Roof_opaque = 0
            A_Roof_window = 0
            for Roof_coordinate in Roof_coordinates:
                A_roof = calc_area_2d(calc_rotate_mat(Roof_coordinate, center[n]))
                A_Roof_opaque += A_roof * (1 - wwr_Roof)
                A_Roof_window += A_roof * wwr_Roof
                Kw1_roof += Uw1_roof * A_roof * (1 - wwr_Roof)
                Kw2_roof += Uw2_roof * A_roof * (1 - wwr_Roof)
                Cw_roof += Cw_roof_coef * A_roof * (1 - wwr_Roof)
                Ue_roof = Exterior_convective_heat_transfer_coefficient(Roof_coordinate, roof_roughness[-1], v_wind, v_dir, float(T0[11*n+3]), Ta_out, center[n])
                Ui_roof = Interior_convective_heat_transfer_coefficient(Roof_coordinate, float(T0[11*n+1]), float(T0[11*n]), center[n])
                Ke_roof += Ue_roof * A_roof * (1 - wwr_Roof)
                Ki_roof += Ui_roof * A_roof * (1 - wwr_Roof)
                UA_roof += U_window * A_roof * wwr_Roof
                Roof_Direct_Radiation, Roof_window_Direct_Radiation, cosi_roof = Direct_Radiation(h, a, Direct_Normal_Radiation, Roof_coordinate, center[n])
                #SHGC_roof = cosi_roof / (0.735044762230282 * cosi_roof ** 2 - 0.03408181475031143 * cosi_roof + 0.6089337726960152)
                SHGC_roof = 0.352
                #SHGC_roof = cosi_roof / (0.9600129323649899 * cosi_roof ** 2 - 0.32503090839959 * cosi_roof + 0.80110073895986)
                Roof_Diffuse_Radiation = Diffuse_Radiation(Diffuse_Horizontal_Radiation, Roof_coordinate, Direct_Normal_Radiation, h, a, center[n])
                Roof_GroundReflect_Radiation = ground_reflect_radiation(Roof_coordinate, Diffuse_Horizontal_Radiation, Direct_Normal_Radiation, h, ground_reflectance, center[n])
                Roof_Total_Solar_Radiation_intensity = Roof_Direct_Radiation + Roof_Diffuse_Radiation + Roof_GroundReflect_Radiation
                Roof_Window_Total_Solar_Radiation_intensity = Roof_window_Direct_Radiation + Roof_Diffuse_Radiation + Roof_GroundReflect_Radiation
                Qsun1_roof += Roof_Total_Solar_Radiation_intensity * A_roof * (1-wwr_Roof) * Abs_roof[-1]
                Qsun2_roof += Roof_Window_Total_Solar_Radiation_intensity * SHGC_roof * A_roof * wwr_Roof
                Qir_e_roof += Longwave_Radiation(float(T0[11*n+3]), Ta_out, Roof_coordinate, Emittance_roof[-1], T_dew, OpqCld, center[n]) * A_roof * (1-wwr_Roof)





            Kw1_raised_ground = 0
            Kw2_raised_ground = 0
            Cw_raised_ground = 0
            Ki_raised_ground = 0
            Ke_raised_ground = 0
            A_raised_ground_opaque = 0
            Qir_e_raised_ground = 0
            for Raised_Ground_coordinate in Raised_Ground_coordinates:
                A_raised_ground = calc_area_2d(calc_rotate_mat(Raised_Ground_coordinate, center[n]))
                A_raised_ground_opaque += A_raised_ground
                Kw1_raised_ground += Uw1_raised_ground * A_raised_ground
                Kw2_raised_ground += Uw2_raised_ground * A_raised_ground
                Cw_raised_ground += Cw_raised_ground_coef * A_raised_ground
                Ui_raised_ground = Interior_convective_heat_transfer_coefficient(Raised_Ground_coordinate, float(T0[11*n+1]), float(T0[11*n]), center[n])
                Ki_raised_ground += Ui_raised_ground * A_raised_ground
                Ue_raised_ground = Exterior_convective_heat_transfer_coefficient(Raised_Ground_coordinate, raised_ground_roughness[-1], 0, v_dir, float(T0[11*n+3]), Ta_out, center[n])
                Ke_raised_ground += Ue_raised_ground * A_raised_ground
                Qir_e_raised_ground += Longwave_Radiation(float(T0[11*n+3]), Ta_out, Raised_Ground_coordinate, Emittance_raised_ground[-1], T_dew, OpqCld, center[n]) * A_raised_ground



            Kw1_ground = 0
            Kw2_ground = 0
            Cw_ground = 0
            Ki_ground = 0
            Ke_ground = 0
            A_ground_opaque = 0
            Cw_soil = 0
            for Ground_coordinate in Ground_coordinates:
                A_ground = calc_area_2d(calc_rotate_mat(Ground_coordinate, center[n]))
                A_ground_opaque += A_ground
                Kw2_ground += Uw2_ground * A_ground
                Cw_ground += Cw_ground_coef * A_ground
                Ui_ground = Interior_convective_heat_transfer_coefficient(Ground_coordinate, float(T0[11*n+4]), float(T0[11*n]), center[n])
                Ki_ground += Ui_ground * A_ground
                Uw1_ground = Uw1_sgf(Ground_coordinate, d_ew, 1/Uw2_ground, Ui_ground)
                Kw1_ground += Uw1_ground * A_ground
                Cw_soil += 1e6 * A_ground



            Cw_internalmass = 0
            A_internalmass_opaque = 0
            Ki_internalmass = 0
            for InternalMass_coordinate in InternalMass_coordinates:
                A_internalmass = calc_area_2d(calc_rotate_mat(InternalMass_coordinate, center[n]))
                A_internalmass_opaque += A_internalmass
                Ui_internalmass = Interior_convective_heat_transfer_coefficient(InternalMass_coordinate, float(T0[11*n+6]), float(T0[11*n]), center[n])
                Ki_internalmass += Ui_internalmass * A_internalmass
                Cw_internalmass += Cw_internalmass_coef * A_internalmass



            Floor_collect = []
            Floor_ConnectedZones = []
            for Floors in Floor_coordinates:
                Floor_collect.append(Floors[0])
                Floor_ConnectedZones.append(Floors[1])

            Ceiling_collect = []
            Ceiling_ConnectedZones = []
            for Ceilings in Ceiling_coordinates:
                Ceiling_collect.append(Ceilings[0])
                Ceiling_ConnectedZones.append(Ceilings[1])

            InternalWall_collect = []
            InternalWall_ConnectedZones = []
            for InternalWalls in InternalWall_coordinates:
                InternalWall_collect.append(InternalWalls[0])
                InternalWall_ConnectedZones.append(InternalWalls[1])


            A_czs = []
            CZS = []
            Cw_floor = 0
            A_floor_opaque = 0
            Ki_floor = 0
            Kw1_floor = 0
            Kw2_floor = 0
            Ki_floor_czs = 0
            for index, Floor_coordinate in enumerate(Floor_collect):
                A_floor = calc_area_2d(calc_rotate_mat(Floor_coordinate, center[n]))
                A_czs.append(A_floor)
                CZS.append(Floor_ConnectedZones[index])
                A_floor_opaque += A_floor
                Cw_floor += Cw_floor_coef * A_floor
                Ui_floor = Interior_convective_heat_transfer_coefficient(Floor_coordinate, float(T0[11*n+7]), float(T0[11*n]), center[n])
                Ki_floor += Ui_floor * A_floor
                Kw1_floor += Uw1_floor * A_floor
                Kw2_floor += Uw2_floor * A_floor
                Ui_floor_czs = Interior_convective_heat_transfer_coefficient(Floor_coordinate, float(T0[11*n+9]), float(T0[11*n+10]), center[Floor_ConnectedZones[index]-1])
                Ki_floor_czs += Ui_floor_czs * A_floor


            Cw_ceiling = 0
            A_ceiling_opaque = 0
            Ki_ceiling = 0
            Kw1_ceiling = 0
            Kw2_ceiling = 0
            Ki_ceiling_czs = 0
            for index, Ceiling_coordinate in enumerate(Ceiling_collect):
                A_ceiling = calc_area_2d(calc_rotate_mat(Ceiling_coordinate, center[n]))
                A_czs.append(A_ceiling)
                CZS.append(Ceiling_ConnectedZones[index])
                A_ceiling_opaque += A_ceiling
                Cw_ceiling += Cw_ceiling_coef * A_ceiling
                Ui_ceiling = Interior_convective_heat_transfer_coefficient(Ceiling_coordinate, float(T0[11*n+7]), float(T0[11*n]), center[n])
                Ki_ceiling += Ui_ceiling * A_ceiling
                Kw1_ceiling += Uw1_ceiling * A_ceiling
                Kw2_ceiling += Uw2_ceiling * A_ceiling
                Ui_ceiling_czs = Interior_convective_heat_transfer_coefficient(Ceiling_coordinate, float(T0[11*n+9]), float(T0[11*n+10]), center[Ceiling_ConnectedZones[index]-1])
                Ki_ceiling_czs += Ui_ceiling_czs * A_ceiling


            Cw_internalwall = 0
            A_internalwall_opaque = 0
            Ki_internalwall = 0
            Kw1_internalwall = 0
            Kw2_internalwall = 0
            Ki_internalwall_czs = 0
            for index, InternalWall_coordinate in enumerate(InternalWall_collect):
                A_internalwall = calc_area_2d(calc_rotate_mat(InternalWall_coordinate, center[n]))
                A_czs.append(A_internalwall)
                CZS.append(InternalWall_ConnectedZones[index])
                A_internalwall_opaque += A_internalwall
                Cw_internalwall += Cw_internalwall_coef * A_internalwall
                Ui_internalwall = Interior_convective_heat_transfer_coefficient(InternalWall_coordinate, float(T0[11*n+7]), float(T0[11*n]), center[n])
                Ki_internalwall += Ui_internalwall * A_internalwall
                Kw1_internalwall += Uw1_internalwall * A_internalwall
                Kw2_internalwall += Uw2_internalwall * A_internalwall
                Ui_internalwall_czs = Interior_convective_heat_transfer_coefficient(InternalWall_coordinate, float(T0[11*n+9]), float(T0[11*n+10]), center[InternalWall_ConnectedZones[index]-1])
                Ki_internalwall_czs += Ui_internalwall_czs * A_internalwall



            Ki_etc = Ki_exteriorwall + Ki_roof + Ki_raised_ground
            Kw2_etc = Kw2_exteriorwall + Kw2_roof + Kw2_raised_ground
            Cw_etc = Cw_exteriorwall + Cw_roof + Cw_raised_ground
            Kw1_etc = Kw1_exteriorwall + Kw1_roof + Kw1_raised_ground
            Ke_etc = Ke_exteriorwall + Ke_roof + Ke_raised_ground
            Qsun1_etc = Qsun1_exteriorwall + Qsun1_roof
            Qir_e_etc = Qir_e_exteriorwall + Qir_e_roof + Qir_e_raised_ground

            Qsun2 = Qsun2_exteriorwall + Qsun2_roof

            Cw_sgf = Cw_ground + Cw_soil

            Ki_itc = Ki_floor + Ki_ceiling + Ki_internalwall
            Kw2_itc = Kw2_floor + Kw2_ceiling + Kw2_internalwall
            Cw_itc = Cw_floor + Cw_ceiling + Cw_internalwall
            Kw1_itc = Kw1_floor + Kw1_ceiling + Kw1_internalwall
            Ki_czs = Ki_floor_czs + Ki_ceiling_czs + Ki_internalwall_czs

            A_etc = A_exteriorwall_opaque + A_Roof_opaque + A_raised_ground_opaque
            A_itc = A_floor_opaque + A_ceiling_opaque + A_internalwall_opaque
            A_window = A_exteriorwall_window + A_Roof_window
            A_total = A_etc + A_itc + A_ground_opaque + A_internalmass_opaque + A_window

            w_etc = A_etc/A_total
            w_itc = A_itc/A_total
            w_sgf = A_ground_opaque/A_total
            w_im = A_internalmass_opaque/A_total
            w_window = A_window/A_total


            if w_etc != 0:
                Emittance_etc = (Emittance_exteriorwall[0] * A_exteriorwall_opaque + Emittance_roof[0] * A_Roof_opaque +
                                 Emittance_raised_ground[0] * A_raised_ground_opaque) / A_etc
            else:
                Emittance_etc = 0

            if w_itc != 0:
                Emittance_itc = (Emittance_floor[0] * A_floor_opaque + Emittance_ceiling[0] * A_ceiling_opaque +
                                 Emittance_internalwall[0] * A_internalwall_opaque) / A_itc
            else:
                Emittance_itc = 0



            Qir_i_etc = Emittance_etc * 5.67e-8 * A_etc * (w_sgf*((T0[11*n+4]+273.15)**4-(T0[11*n+1]+273.15)**4)+w_im*((T0[11*n+6]+273.15)**4-(T0[11*n+1]+273.15)**4)+w_itc*((T0[11*n+7]+273.15)**4-(T0[11*n+1]+273.15)**4))
            Qir_i_sgf = Emittance_ground[0] * 5.67e-8 * A_ground_opaque * (w_etc*((T0[11*n+1]+273.15)**4-(T0[11*n+4]+273.15)**4)+w_im*((T0[11*n+6]+273.15)**4-(T0[11*n+4]+273.15)**4)+w_itc*((T0[11*n+7]+273.15)**4-(T0[11*n+4]+273.15)**4))
            Qir_im = Emittance_internalmass[0] * 5.67e-8 * A_internalmass_opaque * (w_etc*((T0[11*n+1]+273.15)**4-(T0[11*n+6]+273.15)**4)+w_sgf*((T0[11*n+4]+273.15)**4-(T0[11*n+6]+273.15)**4)+w_itc*((T0[11*n+7]+273.15)**4-(T0[11*n+6]+273.15)**4))
            Qir_itc = Emittance_itc * 5.67e-8 * A_itc * (w_etc*((T0[11*n+1]+273.15)**4-(T0[11*n+7]+273.15)**4)+w_sgf*((T0[11*n+4]+273.15)**4-(T0[11*n+7]+273.15)**4)+w_im*((T0[11*n+6]+273.15)**4-(T0[11*n+7]+273.15)**4))

            # Calculate the volume of the thermal zone
            Vma = thermalzone_body_volume(Roof_coordinates, Ground_coordinates, Raised_Ground_coordinates, Floor_collect, Ceiling_collect, center[n])

            Cin = Cpma * Rioma * Vma  # J/K, heat capacity of indoor air

            # Define internal heat sources: sensible and latent heat from people, lighting, and equipment
            people_density = 10  # m²/person
            A_ThermalZone = A_ground_opaque + A_raised_ground_opaque + A_floor_opaque  # Thermal zone floor area
            people_number = A_ThermalZone / people_density * people_inroom_rate  # Number of people in thermal zone
            Heat_Perpeople = 134  # Heat dissipation per person in W/person (input from frontend)
            Heat_people_total = people_number * Heat_Perpeople  # Total heat from people in W
            Heat_Light_Perarea = 9   # W/m², lighting load intensity
            Heat_Light_total = Heat_Light_Perarea * A_ThermalZone * Light_use_rate  # Total heat from lighting in W
            Heat_Equip_Perarea = 15   # W/m², equipment load intensity
            Heat_Equip_total = Heat_Equip_Perarea * A_ThermalZone * Eqip_use_rate  # Total heat from equipment in W

            # Define mechanical ventilation rate (input from frontend)
            mec_vent_perpeople = 25.4844  # m³/h/person, fresh air demand per person
            mec_vent_total = mec_vent_perpeople * people_number  # Total mechanical ventilation in m³/h

            n_mec = mec_vent_total / Vma  # Mechanical air change rate (1/h), 0 if system is off
            heat_recovery = 0  # Heat recovery efficiency of ventilation system (input from frontend)

            # Define infiltration rate (input from frontend)
            ach = 0.5  # Air changes per hour (1/h)

            # Define natural ventilation rate (input from frontend), related to door/window openings
            n_vent = 0  # Air changes per hour (1/h)

            # Total internal heat sources (people, lighting, equipment)
            Sr = Heat_people_total * 0.3  # Latent heat (W)
            Sc = Heat_people_total * 0.7 + Heat_Light_total + Heat_Equip_total  # Sensible heat (W)

            # Calculate total UA value (includes envelope conduction and air exchange losses)
            UA = UA_exteriorwall + UA_roof + (ach + n_vent + n_mec*(1-heat_recovery)*mec_vent_status) * Vma * Rioma * Cpma / 3600


            A[11*n][11*n] = Cin / dt + UA + Ki_etc + Ki_ground + Ki_internalmass + Ki_itc
            A[11*n][11*n+1] = -Ki_etc
            A[11*n][11*n+4] = -Ki_ground
            A[11*n][11*n+6] = -Ki_internalmass
            A[11*n][11*n+7] = -Ki_itc
            b[11*n] = Cin / dt * T0[11*n] + UA * Ta_out + Sc + Qsun2 * w_window

            A[11*n+1][11*n] = -Ki_etc
            A[11*n+1][11*n+1] = Kw2_etc + Ki_etc
            A[11*n+1][11*n+2] = -Kw2_etc
            b[11*n+1] = (Qsun2 + Sr) * w_etc + Qir_i_etc

            A[11*n+2][11*n+1] = -Kw2_etc
            A[11*n+2][11*n+2] = Cw_etc / dt + Kw1_etc + Kw2_etc
            A[11*n+2][11*n+3] = -Kw1_etc
            b[11*n+2] = Cw_etc / dt * T0[11*n+2]


            A[11*n+3][11*n+2] = -Kw1_etc
            A[11*n+3][11*n+3] = Kw1_etc + Ke_etc
            b[11*n+3] = Ke_etc * Ta_out + Qsun1_etc + Qir_e_etc

            A[11*n+4][11*n] = -Ki_ground
            A[11*n+4][11*n+4] = Kw2_ground + Ki_ground
            A[11*n+4][11*n+5] = -Kw2_ground
            b[11*n+4] = (Qsun2 + Sr) * w_sgf + Qir_i_sgf

            A[11*n+5][11*n+4] = -Kw2_ground
            A[11*n+5][11*n+5] = Cw_sgf / dt + Kw1_ground + Kw2_ground
            b[11*n+5] = Cw_sgf / dt * T0[11*n+5] + Kw1_ground * Deep_GroundTemp


            A[11*n+6][11*n] = -Ki_internalmass
            A[11*n+6][11*n+6] = Cw_internalmass / dt + Ki_internalmass
            b[11*n+6] = Cw_internalmass / dt * T0[11*n+6] + (Qsun2 + Sr) * w_im + Qir_im

            A[11*n+7][11*n] = -Ki_itc
            A[11*n+7][11*n+7] = Ki_itc + Kw2_itc
            A[11*n+7][11*n+8] = -Kw2_itc
            b[11*n+7] = (Qsun2 + Sr) * w_itc + Qir_itc

            A[11*n+8][11*n+7] = -Kw2_itc
            A[11*n+8][11*n+8] = Cw_itc / dt + Kw1_itc + Kw2_itc
            A[11*n+8][11*n+9] = -Kw1_itc
            b[11*n+8] = Cw_itc / dt * T0[11*n+8]


            A[11*n+9][11*n+8] = -Kw1_itc
            A[11*n+9][11*n+9] = Kw1_itc + Ki_czs
            A[11*n+9][11*n+10] = -Ki_czs

            Qsun2_zones_collect.append(Qsun2)
            Sr_zones_collect.append(Sr)
            Qir_itc_zones_collect.append(float(Qir_itc))
            A_czs_zones_collect.append(A_czs)
            CZS_zones_collect.append(CZS)
            A_total_zones_collect.append(A_total)
            A_itc_area_zones_collect.append(A_itc)
            A_floor_area_zones_collect.append(A_ThermalZone)

        Heatloads = np.zeros((nZones, 1))
        Coolingloads = np.zeros((nZones, 1))


        for n in range(0, nZones):
            Connect_zones = CZS_zones_collect[n]
            A_Connect_zones = A_czs_zones_collect[n]
            Qsun2_czs = 0
            Sr_czs = 0
            Qir_czs = 0
            A_itc_sum = 0
            for index, zone in enumerate(Connect_zones):
                Qsun2_czs += Qsun2_zones_collect[zone-1] * A_Connect_zones[index] / A_total_zones_collect[zone-1]
                Sr_czs += Sr_zones_collect[zone-1] * A_Connect_zones[index] / A_total_zones_collect[zone-1]
                if A_itc_area_zones_collect[zone-1] == 0:
                    Qir_czs = 0
                else:
                    Qir_czs += Qir_itc_zones_collect[zone-1] * A_Connect_zones[index] / A_itc_area_zones_collect[zone-1]
                A_itc_sum += A_Connect_zones[index]
                A[11*n+10][11*(zone-1)] = -A_Connect_zones[index]
            b[11*n+9] = Qsun2_czs + Sr_czs + Qir_czs
            A[11*n+10][11*n+10] = A_itc_sum



        A_zeros_row = np.where(~A.any(axis=1))[0]
        A_no_zeros = []
        for i in range(len(A)):
            if i not in A_zeros_row:
                A_no_zeros.append(i)
        A_zeros_row_sorted = sorted(A_zeros_row, reverse=True)
        A_no_zeros_sorted = sorted(A_no_zeros, reverse=False)
        A_transform = A.copy()
        b_transform = b.copy()
        for i in A_zeros_row_sorted:
            A_delete_zeros_row = np.delete(A_transform, i, 0)
            A_transform = np.delete(A_delete_zeros_row, i, 1)
            b_transform = np.delete(b_transform, i, 0)

        L = LU(A_transform)[0]
        U = LU(A_transform)[1]
        Ux = np.mat(L).I * np.mat(b_transform)
        T_transform = np.mat(U).I * np.mat(Ux)
        for i, Temperature in enumerate(T_transform):
            T[A_no_zeros_sorted[i]] = Temperature



        for n in range(nZones):
           if AC_status == 1:
                if T[11*n] < T_indoor_set:
                    Tain_set[11*n] = T_indoor_set
                else:
                    Tain_set[11*n] = T[11*n]
           elif AC_status == 2:
                if T[11*n] > T_indoor_set:
                    Tain_set[11*n] = T_indoor_set
                else:
                    Tain_set[11*n] = T[11*n]
           else:
               Tain_set[11 * n] = T[11 * n]


        Tain_location = []
        for n in range(nZones):
            Tain_location.append(11*n)
        delete_number = []
        delete_number.extend(Tain_location)
        delete_number.extend(A_zeros_row)
        delete_number_sorted = sorted(delete_number, reverse=True)

        b_new = np.mat(b) - np.mat(A) * np.mat(Tain_set)
        A_new = A.copy()
        for i in delete_number_sorted:
            A_delete_row = np.delete(A_new, i, 0)
            A_new = np.delete(A_delete_row, i, 1)
            b_new = np.delete(b_new, i, 0)

        L_envelope = LU(A_new)[0]
        U_envelope = LU(A_new)[1]
        Ux_envelope = np.mat(L_envelope).I * np.mat(b_new)
        T_envelope = np.mat(U_envelope).I * np.mat(Ux_envelope)

        A_envelop_location = []
        for i in range(len(T)):
            if i not in delete_number:
                A_envelop_location.append(i)
        A_envelop_location_sorted = sorted(A_envelop_location, reverse=False)

        for i, Temperature in enumerate(T_envelope):
            T_new[A_envelop_location[i]] = Temperature

        T_new = np.mat(T_new) + np.mat(Tain_set)
        Loads = np.mat(A) * np.mat(T_new) - np.mat(b)


        for n in range(nZones):
            if AC_status == 1:
                if Loads[11*n] >= 0:
                    Heatloads[n] = Loads[11*n]
                else:
                    Heatloads[n] = 0
            elif AC_status == 2:
                if Loads[11*n] <= 0:
                    Coolingloads[n] = -Loads[11*n]
                else:
                    Coolingloads[n] = 0
            else:
                Heatloads[n] = 0
                Coolingloads[n] = 0

            Energy_Heat[n] = Heatloads[n] / COP_Heat     # Energy heating demand
            Energy_Cooling[n] = Coolingloads[n] / COP_Cooling  # Energy cooling demand
            Energy_Heat_Perarea[n] = Energy_Heat[n]/A_floor_area_zones_collect[n]  # Energy heating demand per area
            Energy_Cooling_Perarea[n] = Energy_Cooling[n]/A_floor_area_zones_collect[n]  # Energy cooling demand per area


        T = T_new.copy()



        Load = np.concatenate((Heatloads, Coolingloads), axis=0)
        Energy = np.concatenate((Energy_Heat, Energy_Cooling), axis=0)
        Energy_Perarea = np.concatenate((Energy_Heat_Perarea, Energy_Cooling_Perarea), axis=0)




        time = time + dt
        print("outiter = %4d, time=%8.5f" % (istep, time))


        temp = np.append(temp, T, axis=1)
        Energy_Loads = np.append(Energy_Loads, Load, axis=1)
        Energy_Consumption = np.append(Energy_Consumption, Energy, axis=1)
        Energy_Consumption_Perarea = np.append(Energy_Consumption_Perarea, Energy_Perarea, axis=1)


    data_Temp = pd.DataFrame(temp)
    data_Energy_Loads = pd.DataFrame(Energy_Loads)
    data_Energy_Consumption = pd.DataFrame(Energy_Consumption)
    data_Energy_Consumption_Perarea = pd.DataFrame(Energy_Consumption_Perarea)


    writer1 = pd.ExcelWriter('Temp.xlsx')
    writer2 = pd.ExcelWriter('Energy_Loads.xlsx')
    writer3 = pd.ExcelWriter('Energy_Consumption.xlsx')
    writer4 = pd.ExcelWriter('Energy_Consumption_Perarea.xlsx')

    data_Temp.to_excel(writer1, 'Page_1', float_format='%.5f')
    data_Energy_Loads.to_excel(writer2, 'Page_1', float_format='%.5f')
    data_Energy_Consumption.to_excel(writer3, 'Page_1', float_format='%.5f')
    data_Energy_Consumption_Perarea.to_excel(writer4, 'Page_1', float_format='%.5f')

    writer1.save()
    writer2.save()
    writer3.save()
    writer4.save()

    writer1.close()
    writer2.close()
    writer3.close()
    writer4.close()
