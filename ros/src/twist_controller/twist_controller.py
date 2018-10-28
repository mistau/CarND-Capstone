import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit, brake_deadband, vehicle_mass, wheel_radius):
    	# Create controller for steering
        self.yaw_controller = YawController( wheel_base=wheel_base,
                                             steer_ratio=steer_ratio,
                                             min_speed=min_speed,
                                             max_lat_accel=max_lat_accel,
                                             max_steer_angle=max_steer_angle )

        # Create controller for throttle/brake
#        self.pid = PID( kp=5, ki=0.5, kd=0.5, mn=decel_limit, mx=accel_limit )
        self.pid = PID( kp=0.3, ki=0.1, kd=2.5, mn=decel_limit, mx=accel_limit )

        # Low pass filters for these 2 controllers
        # Cutoff frequence = 1/(2pi * tau)
        # ts = sample time
#        self.s_lpf = LowPassFilter( tau = 3, ts = 1 )
#        self.t_lpf = LowPassFilter( tau = 3, ts = 1 )
#        self.s_lpf = LowPassFilter( tau = 0.5, ts = 0.02 )
#        self.t_lpf = LowPassFilter( tau = 0.5, ts = 0.02 )
        self.vel_lpf = LowPassFilter( tau = 0.5, ts = 1 )

        # Need these values for calculation of brake torque
        self.brake_deadband = brake_deadband
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.decel_limit = decel_limit

    def reset(self):
        self.pid.reset()

    def control(self, twist_cmd, current_velocity, delta_time):
    	current_linear_velocity = current_velocity.twist.linear.x
    	current_linear_velocity = self.vel_lpf.filt(current_linear_velocity)
#        rospy.logwarn("Current Linear velocity: {0}".format(current_linear_velocity))

        desired_linear_velocity = twist_cmd.twist.linear.x
#        rospy.logwarn("Desired Linear vel: {0}".format(desired_linear_velocity))

        desired_angular_velocity = twist_cmd.twist.angular.z
#        rospy.logwarn("Desired Angular vel: {0}".format(desired_angular_velocity))

        velocity_error = desired_linear_velocity - current_linear_velocity
#        rospy.logwarn("Velocity Error: {0}".format(velocity_error))

        # Get new steering angle from yaw controller
        steering = self.yaw_controller.get_steering(desired_linear_velocity, desired_angular_velocity, current_linear_velocity)
#        steering = self.s_lpf.filt(steering)

        # Get new acceleration from throttle controller
        throttle = self.pid.step(velocity_error, delta_time)
        brake = 0

#        rospy.logwarn("Throttle: {0}".format(throttle))
#        rospy.logwarn("Brake: {0}".format(brake))
#        rospy.logwarn("Steering: {0}".format(steering))

        if desired_linear_velocity == 0. and current_linear_velocity < 0.1:
        	throttle = 0.0
        	brake = 700

        elif throttle < 0.1 and velocity_error < 0.0:
        	throttle = 0.0
        	decelleration = max(abs(velocity_error), self.decel_limit)
        	brake = abs(decelleration) * self.vehicle_mass * self.wheel_radius

#        rospy.logwarn("New Throttle: {0}".format(throttle))
#        rospy.logwarn("New Brake: {0}".format(brake))
#        rospy.logwarn("New Steering: {0}".format(steering))

        return throttle, brake, steering
