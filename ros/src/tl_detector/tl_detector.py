#!/usr/bin/env python
import rospy, numpy, tf, cv2, yaml, datetime
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial import KDTree
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 3
IMAGE_COUNT_THRESHOLD = 3
TL_DETECTOR_DEBUG = False

class TLDetector(object):
    def __init__(self):
        self.initialised = False
        rospy.init_node('tl_detector', log_level=rospy.DEBUG)

        self.light_enum = {
            4:'UNKNOWN',
            2:'GREEN',
            1:'YELLOW',
            0:'RED',
        }

        self.pose = None # current pose of car
        self.base_waypoints = None # map waypoints collected at start of sim
        self.waypoints_2d = None # map waypoints reduced to (x,y) coordinates
        self.waypoint_tree = None # KDTree of map waypoints
        self.camera_image = None # current camera image for classifier
        self.lights = [] # list of traffic lights

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic
        light in 3D map space and helps you acquire an accurate ground truth
        data source for the traffic light classifier by sending the current
        color state of all traffic lights in the simulator. When testing on the
        vehicle, the color state will not be available. You'll need to rely on
        the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                                self.traffic_cb)

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint',
                                                      Int32, queue_size=1)

        self.bridge = CvBridge()

        # load appropriate frozen model
        self.is_site = self.config["is_site"]
        if not self.is_site:
            self.light_classifier = TLClassifier(True)
            rospy.loginfo("Using neuronal network trained with simulator data")
        else:
            self.light_classifier = TLClassifier(False)
            rospy.loginfo("Using neuronal network trained with camera images")

        self.listener = tf.TransformListener()

        self.light_state = TrafficLight.UNKNOWN
        self.light_wp = -1
        self.state_count = 0
        self.image_count = 0

        if TL_DETECTOR_DEBUG is True:
            rospy.logwarn("Using Basic TLDetector")
        else:
            rospy.loginfo("Using TLClassifier")

        self.initialised = True
        rospy.spin()


    def pose_cb(self, msg):
        """
        Callback function to consume PoseStamped objects from /current_pose
        topic.
        """
        self.pose = msg

    # TODO merge with 'waypoint_updater' node?
    def waypoints_cb(self, waypoints):
        """
        Callback function to consume Lane objects from /base_waypoints topic.
        """
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        """
        Callback function to consume Image objects from /image_color topic.
        """
        self.lights = msg.lights

    def log_tl_state_change(self, wp, old_state, new_state):
        """
        Logs details about the next traffic light to the console, including:
            TL<IDX>, where IDX is the x coordinate of the next traffic light
            The state of the traffic light before it changed
            The state of the traffic light after it changed
        """
        if TL_DETECTOR_DEBUG is True:
            rospy.logdebug(
                "TL{} changed from {} to {}".format(
                    wp, self.light_enum[old_state], self.light_enum[new_state]
                )
            )

    def image_cb(self, msg):
        """
        Identifies red lights in the incoming camera image and publishes the
        index of the waypoint closest to the red light's stop line to
        /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        if self.initialised is True:
            if self.image_count < IMAGE_COUNT_THRESHOLD:
                self.image_count += 1
                self.has_image = False
                new_light_wp = self.light_wp
                new_state = self.light_state

            else:
                self.image_count = 0
                self.has_image = True
                self.camera_image = msg
                new_light_wp, new_state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency. Each predicted state
            has to occur `STATE_COUNT_THRESHOLD` number of times until we start
            using it. Otherwise the previous stable state is used.
            '''
            self.light_wp = new_light_wp

            if self.light_state != new_state:
                self.log_tl_state_change(self.light_wp, self.light_state, new_state)
                self.light_state = new_state
                self.state_count = 0
            else:
                if self.state_count >= STATE_COUNT_THRESHOLD:
                    self.light_wp = self.light_wp if self.light_state == TrafficLight.RED else -1
                self.upcoming_red_light_pub.publish(Int32(self.light_wp))

            if self.has_image is True:
                self.state_count += 1

    # TODO merge with 'waypoint_updater' node?
    def get_closest_waypoint(self, x, y):
        """
        Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

        Args:
            x (Int): x coordinate of position to match a waypoint to
            y (Int): y coordinate of position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.base_waypoints
        """

        closest_wp_idx = self.waypoint_tree.query([x, y], 1)[1]
        # return closest_wp_idx

        # check if closest point is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_wp_idx]
        prev_coord = self.waypoints_2d[closest_wp_idx-1]

        # hyperplane equation through closest_coords
        cl_vect = numpy.array(closest_coord)
        prev_vect = numpy.array(prev_coord)
        pos_vect = numpy.array([x, y])

        val = numpy.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val>0:
            closest_wp_idx = (closest_wp_idx+1) % len(self.waypoints_2d)
        return closest_wp_idx

    def get_light_state(self, light):
        """
        Determines the current color of the traffic light.

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            or None if no camera image exists
        """

        if TL_DETECTOR_DEBUG is True:
            return light.state

        else:
            if self.has_image is False:
                return None

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its
        location and color

        Returns:
            int: Index of waypoint closest to the upcoming stop line for a
            traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        closest_light = None # Closest traffic light
        line_wp_idx = None # Stop line waypoint index for above traffic light

        # List of positions that correspond to the line to stop in front of for
        # a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose is not None and self.waypoint_tree is not None:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,
                                                   self.pose.pose.position.y)

            # Find the closest visible traffic light (if one exists)
            diff = len(self.base_waypoints.waypoints) # start with longest distance
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_index = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_index - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_index

        if closest_light is not None:
            state = self.get_light_state(closest_light)
            if state is not None:
                return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
