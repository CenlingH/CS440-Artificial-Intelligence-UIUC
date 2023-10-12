# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

# 真的好难啊
"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
# from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

    Args:
        alien (Alien): Instance of Alien class that will be navigating our map
        walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                     [(startx, starty, endx, endx), ...]

    Return:
        True if touched, False if not
    """
    centroid = alien.get_centroid()
    shape = alien.get_shape()

    def check_segment_intersection(segment1, segment2):
        return do_segments_intersect(segment1, segment2)

    if shape == 'Ball':
        for i in walls:
            s1 = (i[0], i[1])
            s2 = (i[2], i[3])
            if point_segment_distance(centroid, (s1, s2)) <= alien.get_width():
                return True
    else:
        head, tail = sorted(alien.get_head_and_tail(),
                            key=lambda point: point[0])
        width = alien.get_width()

        if shape == "Horizontal":
            x1 = (head[0], head[1] - width)
            y1 = (tail[0], tail[1] - width)
            x2 = (head[0], head[1] + width)
            y2 = (tail[0], tail[1] + width)
        elif shape == "Vertical":
            x1 = (head[0] - width, head[1])
            y1 = (tail[0] - width, tail[1])
            x2 = (head[0] + width, head[1])
            y2 = (tail[0] + width, tail[1])

        for i in walls:
            s1 = (i[0], i[1])
            s2 = (i[2], i[3])
            if check_segment_intersection((x1, y1), (s1, s2)) or check_segment_intersection((x2, y2), (s1, s2)) or \
                    point_segment_distance((head[0], head[1]), (s1, s2)) <= width or \
                    point_segment_distance((tail[0], tail[1]), (s1, s2)) <= width:
                return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window
            Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    w_w = window[0]
    w_h = window[1]
    w1 = [(0, 0, 0, w_h),
          (w_w, 0, w_w, w_h)]
    w2 = [(0, 0, w_w, 0),
          (0, w_h, w_w, w_h)]
    center = alien.get_centroid()
    shape = alien.get_shape()
    for i in w1:
        start = (i[0], i[1])
        end = (i[2], i[3])
        if shape == "Horizontal":
            max_dis = alien.get_length() / 2 + alien.get_width()
        else:
            max_dis = alien.get_width()
        if point_segment_distance(center, (start, end)) <= max_dis:
            return False
    for i in w2:
        start = (i[0], i[1])
        end = (i[2], i[3])
        if shape == "Vertical":
            max_dis = alien.get_length() / 2 + alien.get_width()
        else:
            max_dis = alien.get_width()
        if point_segment_distance(center, (start, end)) <= max_dis:
            return False
    return True


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if does_alien_touch_wall(alien, walls):
        return True
    centroid = alien.get_centroid()
    shape = alien.get_shape()
    if centroid[0] == waypoint[0]:
        if centroid[1] == waypoint[1]:
            alien.set_alien_pos(waypoint)
            if does_alien_touch_wall(alien, walls):
                return True
            return False
        else:
            dir1 = 1
    elif centroid[1] == waypoint[1]:
        dir1 = 0
    else:
        dir1 = 2

    width = alien.get_width()
    length = alien.get_length() / 2 + alien.get_width()
    if dir1 == 1 and shape == "Vertical":
        p = [(centroid[0] - width, centroid[1]),
             (centroid[0] + width, centroid[1]),
             (waypoint[0] + width, waypoint[1]),
             (waypoint[0] - width, waypoint[1])]

    if dir1 == 0 and shape == "Horizontal":
        p = [(centroid[0], centroid[1] - width),
             (centroid[0], centroid[1] + width),
             (waypoint[0], waypoint[1] + width),
             (waypoint[0], waypoint[1] - width)]
    else:
        if shape == "Ball":
            vector = [waypoint[0] - centroid[0], waypoint[1] - centroid[1]]
            if np.all(vector == 0) or np.all([1, 0] == 0):
                c1 = 0
            else:
                c1 = np.dot(
                    vector, [1, 0]) / (np.linalg.norm(vector) * np.linalg.norm([1, 0]))
            c2 = np.sqrt(1 - c1 * c1)
            p = [
                (centroid[0] - c2*width, centroid[1] - width * c1),
                (waypoint[0] - c2*width, waypoint[1] - width * c1),
                (waypoint[0] + c2*width, waypoint[1] + width * c1),
                (centroid[0] + c2*width, centroid[1] + width * c1)
            ]
        elif shape == "Vertical":
            p = [
                (centroid[0],  centroid[1] - length),
                (waypoint[0], waypoint[1] - length),
                (waypoint[0], waypoint[1] + length),
                (centroid[0],  centroid[1] + length)
            ]
        elif shape == "Horizontal":
            p = [
                (centroid[0] + length,  centroid[1]),
                (waypoint[0] + length, waypoint[1]),
                (waypoint[0] - length, waypoint[1]),
                (centroid[0] - length,  centroid[1])
            ]
    edges = [
        (p[0], p[1]),
        (p[1], p[2]),
        (p[2], p[3]),
        (p[3], p[0])
    ]
    for i in walls:
        if is_point_in_polygon((i[0], i[1]), p):
            return True
        elif is_point_in_polygon((i[2], i[3]), p):
            return True
        else:
            for j in edges:
                if do_segments_intersect(((i[0], i[1]), (i[2], i[3])), j):
                    return True
    alien.set_alien_pos(waypoint)
    if does_alien_touch_wall(alien, walls):
        alien.set_alien_pos(centroid)
        return True
    alien.set_alien_pos(centroid)
    return False

# ok


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

    Args:
        p: A tuple (x, y) of the coordinates of the point.
        s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

    Return:
        Euclidean distance from the point to the line segment.
    """
    # Calculate the differences in x and y coordinates between the segment endpoints
    delta_x = s[1][0] - s[0][0]
    delta_y = s[1][1] - s[0][1]

    # Calculate the length of the line segment using the Pythagorean theorem
    len_s = np.sqrt(delta_x**2 + delta_y**2)

    # Calculate the length of the vector from the point to the start of the segment
    len_p = np.sqrt((p[0]-s[0][0])**2 + (p[1]-s[0][1])**2)

    # Calculate the dot product between the vector from the point to the start of the segment and the segment itself
    dot_product = (p[0]-s[0][0])*delta_x + (p[1]-s[0][1])*delta_y

    # If the dot product is less than 0, the closest point is the start of the segment
    if dot_product < 0:
        return np.sqrt((p[0]-s[0][0])**2 + (p[1]-s[0][1])**2)

    # If the dot product is greater than the square of the length of the segment, the closest point is the end of the segment
    elif dot_product > len_s**2:
        return np.sqrt((p[0]-s[1][0])**2 + (p[1]-s[1][1])**2)

    # Otherwise, the closest point is somewhere along the segment, and we calculate the perpendicular distance (touying)
    else:
        touying = dot_product/len_s
        return np.sqrt(len_p**2 - touying**2)


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    A, B = s1
    C, D = s2

    def on_segment(p, q, r):
        """Check if point q lies on line segment pr."""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def ori(p, q, r):
        """Find the orientation of triplet (p, q, r)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    # Check if the segments have any endpoints in common
    if A == C or A == D or B == C or B == D:
        return True

    # Check for orientation of points
    o1 = ori(A, B, C)
    o2 = ori(A, B, D)
    o3 = ori(C, D, A)
    o4 = ori(C, D, B)

    # Check for general case
    if o1 != o2 and o3 != o4:
        return True

    # Check for special cases where the segments overlap
    if o1 == 0 and on_segment(A, C, B):
        return True
    if o2 == 0 and on_segment(A, D, B):
        return True
    if o3 == 0 and on_segment(C, A, D):
        return True
    if o4 == 0 and on_segment(C, B, D):
        return True

    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    '''
    segment_distance(segment1, segment2): Compute the Euclidean distance between two line segments, defined as the shortest distance between any pair of points on the two segments.
    '''
    if do_segments_intersect(s1, s2):
        return 0
    else:
        return min(point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1))


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    x, y = point
    x1 = polygon[0][0]
    y1 = polygon[0][1]
    x2 = polygon[1][0]
    y2 = polygon[1][1]
    x3 = polygon[2][0]
    y3 = polygon[2][1]
    x4 = polygon[3][0]
    y4 = polygon[3][1]
    if x == x1 and x == x2 and x == x3 and x == x4:
        if y >= min(y1, y2, y3, y4) and y <= max(y1, y2, y3, y4):
            return True
        else:
            return False
    if y == y1 and y == y2 and y == y3 and y == y4:
        if x >= min(x1, x2, x3, x4) and x <= max(x1, x2, x3, x4):
            return True
        else:
            return False
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / \
                            (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints

    # Here we first test your basic geometry implementation

    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]),
                       (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'

    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]),
                         (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'

    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]),
                         (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'

    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'

    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(
                alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.

    alien_ball = Alien((30, 120), [40, 0, 40], [
                       11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [
                       11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [
                       11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(
        centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(),
                (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(),
                (True, False, True))

    print("Geometry tests passed\n")
