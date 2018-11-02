#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 10      # Max. acceptable deceleration is 10m/s/s


class WaypointUpdater(object):
    """
    Provides updated waypoints
    """

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # init member variables
        self.have_waypoints = False
        self.base_lane_wp_len = 0
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = -1
        self.cache_closest_wp_idx = -1
        self.cache_decel_waypoints = None
        self.last_closest_idx = -1
        self.last_base_waypoints = None

        # inputs of the ROS module
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # outputs of the ROS module
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.pose and self.base_lane and self.have_waypoints:
                self.publish_waypoints()

    def get_closest_waypoint_idx(self):
        """
        return index of the waypoint closest to the car's position
        """
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_wp_idx = self.waypoint_tree.query([x,y], 1)[1]

        # check if closest point is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_wp_idx]
        prev_coord = self.waypoints_2d[closest_wp_idx-1]

        # hyperplane equation through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val>0:
            closest_wp_idx = (closest_wp_idx+1) % len(self.waypoints_2d)
        return closest_wp_idx


    def publish_waypoints(self):
        """
        Generate a new list of waypoints and publish
        """
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)


    def generate_lane(self):
        """
        Generate new list of waypoints
        takes into account a detected traffic light
        """
        lane = Lane()

        # \todo MS:check if the statement below is required
        #lane.header = self.base_lane.header

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        
        if closest_idx == self.last_closest_idx:
            base_waypoints = self.last_base_waypoints
        elif self.last_base_waypoints == None:
            base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        else:
            base_waypoints = self.last_base_waypoints
            for i in range(self.last_closest_idx, closest_idx):
                if len(base_waypoints) > 1:
                    base_waypoints.pop(0)
                if i+LOOKAHEAD_WPS < self.base_lane_wp_len:
                    base_waypoints.append( self.base_lane.waypoints[i + LOOKAHEAD_WPS] )
        
        # maintain cache
        self.last_closest_idx = closest_idx
        self.last_base_waypoints = base_waypoints
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            # easy case - continue driving
            lane.waypoints = base_waypoints
        else:
            # red traffic light ahead - we have to break
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane


    def decelerate_waypoints(self, waypoints, closest_idx):
        """
        Calculate list of waypoints for deceleration in front of a red traffic light
        waypoints:   list of waypoints ahead of us (i.e. a part of the waypoint list!
        closest_idx: index of the car's position in the global waypoint list
        """

        # check if we just calculated this!
        if closest_idx == self.cache_closest_wp_idx:
            return self.cache_decel_waypoints

        rospy.loginfo("Calculating deceleration path for idx={0}".format(closest_idx))
        temp = []
        was_zero = False
        for i,wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            if was_zero:
                v = 0
            else:
                stop_idx = max(self.stopline_wp_idx - closest_idx -3, 0) # stop three waypoints before!
                dist = self.distance(waypoints, i, stop_idx)
                v = math.sqrt(2 * MAX_DECEL * (dist/20)**3)
                if v<1.0:
                    v = 0.0
                    was_zero = True

            p.twist.twist.linear.x = min(v, wp.twist.twist.linear.x)
            temp.append(p)

        # cache our result to remove latency
        self.cache_closest_wp_idx = closest_idx
        self.cache_decel_waypoints  = temp

        return temp


    def pose_cb(self, msg):
        """
        Callback: receives pose of the car
        """
        self.pose = msg


    def waypoints_cb(self, waypoints):
        """
        Callback: receives waypoints of track
        Attention: this is only published once, keep the data safe!
        """
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_lane_wp_len = len(self.base_lane.waypoints)
        self.have_waypoints = True
        rospy.loginfo("Received {} waypoints".format(self.base_lane_wp_len))


    def traffic_cb(self, msg):
        """
        Callback: receive the waypoint index of a detected red traffic light
                  -1 means no traffic light
        """
        self.stopline_wp_idx = msg.data
        if self.stopline_wp_idx != -1:
            rospy.loginfo("Red Traffic light detected: {0}".format(self.stopline_wp_idx))


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
